"""Immutable continuation snapshots and independent GitHub save/retry.

No function here invokes generation. Old source/190-response assets are
references only. The runner calls safe_checkpoint while holding all run locks,
after a response is durable or after transport has stopped. Standalone create
obtains those same locks; uncertain reservations are preserved, never cleared.
"""
from __future__ import annotations
import base64
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from urllib.parse import quote
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import uuid

from ..server_v2.checkpoint import PATTERNS, safe_relative, scan_file, restore_archive
from ..server_v2.generation import GenerationError, digest, object_hash, read_json, write_json, utc_now, json_bytes
from ..server_v2.data import sha256_file
from ..server_v2.io_utils import writable

REPOSITORY = 'catlover-bot/Minecraft-view-dataset'


def scan_public_file(path, raw_scope=None):
    """Narrow exception for provider-generated encrypted reasoning, never keys.

    Raw bytes remain unchanged. The artifact digest and provider/model bind the
    only permitted field; its base64url Fernet envelope is checked. All other
    fields and every ordinary file retain the original secret-pattern checks.
    """
    try:
        scan_file(path)
        return []
    except ValueError:
        if raw_scope is None:
            raise
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != raw_scope['sha256'] or raw_scope['provider'] != 'openai':
        raise ValueError('Public-safety scan failed; no eligible encrypted provider field')
    document = read_json(path)
    if document.get('model') not in raw_scope['accepted_models'] or not isinstance(document.get('output'), list):
        raise ValueError('Public-safety scan failed; response identity mismatch')
    normalized = deepcopy(document)
    exceptions = []
    for index, item in enumerate(normalized['output']):
        value = item.get('encrypted_content') if isinstance(item, dict) else None
        if value is None:
            continue
        if (item.get('type') != 'reasoning' or not isinstance(value, str)
                or not re.fullmatch(r'gAAAA[A-Za-z0-9_-]+={0,2}', value)):
            raise ValueError('Unclassified encrypted field; publication blocked')
        try:
            decoded = base64.b64decode(value + '=' * (-len(value) % 4), altchars=b'-_', validate=True)
            timestamp = int.from_bytes(decoded[1:9], 'big')
        except Exception:
            raise ValueError('Invalid encrypted provider envelope') from None
        if len(decoded) < 73 or decoded[0] != 0x80 or not 1262304000 <= timestamp <= 4102444800:
            raise ValueError('Invalid encrypted provider envelope')
        if any(pattern.search(value.encode()) for pattern in PATTERNS):
            exceptions.append({'raw_sha256': raw_scope['sha256'], 'field': f'output[{index}].encrypted_content',
                               'encrypted_field_sha256': digest(value.encode()), 'characters': len(value),
                               'classification': 'verified OpenAI reasoning encrypted_content base64url/Fernet envelope; raw bytes retained'})
        item['encrypted_content'] = 'SCANNED_PROVIDER_ENCRYPTED_REASONING_FIELD'
    if not exceptions or any(pattern.search(json_bytes(normalized)) for pattern in PATTERNS):
        raise ValueError('Secret-like content outside classified provider ciphertext; publication blocked')
    return exceptions


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise GenerationError('checkpoint_regular_nonsymlink_file_required')
    return path


def _archive(path, entries, metadata, raw_scopes):
    """Local variant solely to support the documented per-field scanner rule."""
    if path.exists():
        raise GenerationError('immutable_checkpoint_asset_already_exists')
    if shutil.disk_usage(path.parent).free < 10 * 1024**3:
        raise GenerationError('checkpoint_requires_10GiB_free')
    files, exceptions = [], []
    for name, source in entries:
        safe_relative(name); source = _regular(source)
        exceptions += scan_public_file(source, raw_scopes.get(str(source)))
        files.append({'path': name, 'size': source.stat().st_size, 'sha256': sha256_file(source)})
    if len({r['path'] for r in files}) != len(files):
        raise GenerationError('checkpoint_duplicate_archive_path')
    metadata = {**metadata, 'public_safety_scanner_exceptions': exceptions}
    manifest = {'metadata': metadata, 'files': files}
    mp = path.parent / 'FILES.json'; write_json(mp, manifest); scan_file(mp)
    expected = {row['path']: row for row in files}
    with path.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as archive:
        for name, source in entries + [('FILES.json', mp)]:
            if name != 'FILES.json' and (source.stat().st_size != expected[name]['size'] or sha256_file(source) != expected[name]['sha256']):
                raise GenerationError('checkpoint_source_changed_while_archiving')
            info = archive.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with source.open('rb') as stream:
                archive.addfile(info, stream)
    if path.stat().st_size > 1024**3:
        raise GenerationError('checkpoint_asset_exceeds_1GiB')
    return manifest


def safe_checkpoint(plan, directory, ledger, reason, *, destination=None, github=False):
    """Caller holds parent/run/authority locks and has no active HTTP request."""
    from . import runner
    identity, directory = plan['identity'], Path(directory)
    if not directory.is_dir() or read_json(directory / 'ledger.json') != ledger:
        raise GenerationError('checkpoint_requires_existing_current_ledger')
    suffix = utc_now().replace('-', '').replace(':', '').replace('.', '') + '-' + uuid.uuid4().hex[:8]
    checkpoint_id = f"goal006-{plan['identity_hash'][:12]}-{len(ledger['entries']):04d}-{suffix}"
    destination = runner._checked_path(destination or (Path('/root/minecraft-checkpoints') / checkpoint_id))
    if destination.exists():
        raise GenerationError('new_checkpoint_destination_required')
    # Explicit control/attempt allowlist; never walk /root or credential folders.
    entries = [('continuation/plan.json', _regular(identity['plan_path']))]
    for name in ('approval.json', 'enabled.json', 'ledger.json', 'run.lock'):
        entries.append(('continuation/enabled/' + name, _regular(directory / name)))
    for name in ('evaluation.json', 'summary.json'):
        if (directory / name).exists():
            entries.append(('continuation/enabled/' + name, _regular(directory / name)))
    entries.append(('continuation/authority.json', _regular(identity['authority_path'])))
    revisions = Path(identity['plan_path']).parent / 'pricing-revisions'
    if revisions.exists():
        for index, path in enumerate(sorted(revisions.iterdir()), 1):
            if path.name != f'{index:04d}.json':
                raise GenerationError('checkpoint_pricing_revision_sequence_mismatch')
            entries.append(('continuation/pricing-revisions/' + path.name, _regular(path)))
    raw_scopes = {}
    for entry in ledger['entries']:
        claim = directory / 'reservations' / (entry['attempt_id'] + '.json')
        entries.append(('continuation/enabled/reservations/' + claim.name, _regular(claim)))
        for name, descriptor in entry['artifacts'].items():
            if name not in {'raw.json', 'parsed.json', 'voxels.json'} or descriptor['path'] != f"attempts/{entry['attempt_id']}/{name}":
                raise GenerationError('checkpoint_unapproved_artifact')
            path = _regular(directory / descriptor['path'])
            if sha256_file(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
                raise GenerationError('checkpoint_artifact_hash_mismatch')
            entries.append(('continuation/enabled/' + descriptor['path'], path))
            if name == 'raw.json':
                raw_scopes[str(path)] = {'provider': entry['provider'], 'sha256': descriptor['sha256'],
                    'accepted_models': identity['pricing'][entry['provider']]['accepted_response_model_ids']}
    before = {str(p): sha256_file(p) for _, p in entries}
    destination.mkdir(parents=True)
    environment = {'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
                   'code_commit': identity['commit'], 'implementation_hashes': identity['implementation_hashes'],
                   'environment_variables_recorded': False, 'research_api_requests_sent_by_checkpoint': 0}
    write_json(destination / 'environment.json', environment)
    entries.append(('environment.json', destination / 'environment.json'))
    summary = runner.execution_summary(plan, ledger)
    metadata = {'schema': 'goal006_continuation_point_in_time_v1', 'checkpoint_id': checkpoint_id,
                'created_at_utc': utc_now(), 'reason': reason, 'plan_identity_hash': plan['identity_hash'],
                'canonical_plan_sha256': sha256_file(identity['plan_path']), 'code_commit': identity['commit'],
                'execution': summary, 'request_active': False,
                'source_references': identity['checkpoint_sources'],
                'full_manifest_reference': {'path': identity['manifest_path'], 'sha256': identity['manifest_sha256'],
                                            'identity_hash': identity['manifest_identity_hash']},
                'scope': 'only new continuation originals; prior sources/190 responses reused by asset ID and SHA',
                'research_api_requests_sent_by_checkpoint': 0}
    archive = destination / 'continuation-point-in-time.tar.gz'
    manifest = _archive(archive, entries, metadata, raw_scopes)
    after = {str(p): sha256_file(p) for _, p in entries if str(p) in before}
    if before != after:
        raise GenerationError('checkpoint_originals_changed')
    write_json(destination / 'snapshot.json', {**metadata, 'original_sha256': before,
                                              'archive_sha256': sha256_file(archive), 'file_count': len(manifest['files'])})
    write_json(destination / 'SOURCE_REFERENCE.json', identity['checkpoint_sources'])
    (destination / 'RESTORE.md').write_text('Verify SHA256SUMS and restore with offline_evaluation.server_v2.checkpoint.restore_archive into a NEW directory. Files and links are checked against FILES.json. Fetch older source/190-response assets by SOURCE_REFERENCE.json IDs and SHA; they are not duplicated. The restored plan is a forensic copy: paid activation must remain at the original canonical plan and ledger. GitHub upload retry uses checkpoint-upload and never invokes generation.\n')
    assets = [archive.name, 'FILES.json', 'snapshot.json', 'SOURCE_REFERENCE.json', 'RESTORE.md']
    for name in assets[1:]:
        scan_file(destination / name)
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha256_file(destination / n)}  {n}\n' for n in assets))
    assets.append('SHA256SUMS')
    package = {'version': 'goal006_checkpoint_package_v1', 'repository': REPOSITORY,
               'checkpoint_id': checkpoint_id, 'tag': 'checkpoint-' + checkpoint_id,
               'target_commitish': identity['commit'], 'plan_identity_hash': plan['identity_hash'],
               'archive': archive.name, 'assets': [{'name': n, 'sha256': sha256_file(destination / n),
                                                   'bytes': (destination / n).stat().st_size} for n in assets],
               'status': 'LOCAL_VERIFIED', 'originals_unchanged': True,
               'upload_retry_command': f'python -B -m offline_evaluation.continuation_v1.cli checkpoint-upload --checkpoint {destination}'}
    write_json(destination / 'checkpoint.json', package); scan_file(destination / 'checkpoint.json')
    if github:
        try:
            return {'status': 'GITHUB_VERIFIED', 'checkpoint_directory': str(destination), 'github': upload(destination)}
        except Exception as exc:
            # Saved package survives; generation status is never rewritten.
            return {'status': 'GITHUB_SAVE_FAILED', 'checkpoint_directory': str(destination),
                    'error_type': type(exc).__name__, 'retry_command': package['upload_retry_command'],
                    'generation_ledger_unchanged': True}
    return {'status': 'LOCAL_VERIFIED', 'checkpoint_directory': str(destination), 'archive_sha256': sha256_file(archive)}


def create(plan_path, *, destination=None, github=False):
    from . import plan as planning, runner
    from .aggregate import evaluate_verified
    plan = planning.load_plan(plan_path); identity = plan['identity']
    directory = runner._checked_path(identity['enabled_directory'])
    if not directory.is_dir():
        raise GenerationError('run_not_enabled; checkpoint_cannot_create_enabled')
    with runner._parent_locked(identity), runner.run_lock(directory), runner._authority_locked(identity):
        directory, ledger = runner._load_enabled(plan)
        evaluate_verified(plan, directory, ledger)
        return safe_checkpoint(plan, directory, ledger, 'explicit_safe_checkpoint', destination=destination, github=github)


def _gh(args, *, check=True):
    result = subprocess.run(['gh', *args], capture_output=True, text=True)
    if result.returncode and check:
        raise GenerationError('GitHub_save_failed; local_package_preserved; retry_checkpoint_upload_only')
    return result


def _verify_restored_public(restored):
    """Scan the exact restored bytes and prove copied-plan loading fails."""
    from . import plan as planning
    restored = Path(restored)
    copied_path = restored / 'continuation/plan.json'
    copied = read_json(copied_path)
    scope = {}
    ledger_path = restored / 'continuation/enabled/ledger.json'
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        for entry in ledger['entries']:
            descriptor = entry.get('artifacts', {}).get('raw.json')
            if descriptor:
                scope[str(restored / 'continuation/enabled' / descriptor['path'])] = {
                    'provider': entry['provider'], 'sha256': descriptor['sha256'],
                    'accepted_models': copied['identity']['pricing'][entry['provider']]['accepted_response_model_ids']}
    manifest = read_json(restored / 'FILES.json')
    for row in manifest['files']:
        path = restored / row['path']
        scan_public_file(path, scope.get(str(path)))
    scan_file(restored / 'FILES.json')
    try:
        planning.load_plan(copied_path)
    except GenerationError as exc:
        if 'canonical' not in str(exc) and 'copied' not in str(exc):
            raise GenerationError('restored_plan_rejected_for_unexpected_reason') from None
        return str(exc)
    raise GenerationError('forensic_copy_unexpectedly_accepted_as_live_plan')


def upload(checkpoint_directory):
    """Retry the existing immutable package only; never loads a live run/keys."""
    directory = writable(checkpoint_directory)
    package = read_json(_regular(directory / 'checkpoint.json'))
    if package.get('repository') != REPOSITORY or package.get('version') != 'goal006_checkpoint_package_v1':
        raise GenerationError('unrecognized_checkpoint_package')
    if not re.fullmatch(r'checkpoint-goal006-[A-Za-z0-9-]+', package['tag']) or not re.fullmatch(r'[0-9a-f]{40}', package['target_commitish']):
        raise GenerationError('invalid_checkpoint_release_target')
    for asset in package['assets']:
        safe_relative(asset['name'])
        path = _regular(directory / asset['name'])
        if path.parent != directory or sha256_file(path) != asset['sha256'] or path.stat().st_size != asset['bytes']:
            raise GenerationError('immutable_checkpoint_package_changed')
    stamp = utc_now().replace(':', '').replace('.', '') + '-' + uuid.uuid4().hex[:8]
    attempt = directory / ('upload-' + stamp); attempt.mkdir()
    archive = next(a for a in package['assets'] if a['name'] == package['archive'])
    preflight = attempt / 'preflight-restored'
    restore_archive(directory / archive['name'], preflight, archive['sha256'])
    _verify_restored_public(preflight)
    branch = quote('research/server-only-20261013', safe='')
    ancestry = json.loads(_gh(['api', f"repos/{REPOSITORY}/compare/{package['target_commitish']}...{branch}"]).stdout)
    if ancestry.get('status') not in {'ahead', 'identical'} or ancestry.get('behind_by') != 0:
        raise GenerationError('checkpoint_commit_not_pushed_on_dedicated_branch')
    endpoint = f"repos/{REPOSITORY}/releases/tags/{package['tag']}"
    probe = _gh(['api', endpoint], check=False)
    if probe.returncode:
        if '404' not in probe.stderr:
            raise GenerationError('GitHub_release_lookup_failed; no_creation_after_uncertain_lookup')
        payload = {'tag_name': package['tag'], 'target_commitish': package['target_commitish'], 'draft': True,
                   'name': package['checkpoint_id'], 'body': 'Immutable Goal006 point-in-time checkpoint. Independent unit is building; partial results remain descriptive. Generation is not invoked by checkpoint upload.'}
        write_json(attempt / 'release.json', payload)
        release = json.loads(_gh(['api', '--method', 'POST', f'repos/{REPOSITORY}/releases', '--input', str(attempt / 'release.json')]).stdout)
    else:
        release = json.loads(probe.stdout)
    if release.get('draft') is not True or release.get('target_commitish') != package['target_commitish']:
        raise GenerationError('release_collision_or_not_draft; never_overwrite')
    current = {a['name']: a for a in release.get('assets', [])}
    for asset in package['assets']:
        existing = current.get(asset['name'])
        if existing:
            if existing.get('digest') != 'sha256:' + asset['sha256']:
                raise GenerationError('existing_release_asset_digest_differs; never_overwrite')
        else:
            _gh(['release', 'upload', package['tag'], str(directory / asset['name']), '--repo', REPOSITORY])
    release = json.loads(_gh(['api', f"repos/{REPOSITORY}/releases/{release['id']}"]).stdout)
    current = {a['name']: a for a in release['assets']}
    download = attempt / 'redownload'; download.mkdir()
    for asset in package['assets']:
        live = current.get(asset['name'])
        if not live or live.get('digest') != 'sha256:' + asset['sha256'] or live['size'] != asset['bytes']:
            raise GenerationError('GitHub_asset_digest_or_size_mismatch')
        _gh(['release', 'download', package['tag'], '--repo', REPOSITORY, '--pattern', asset['name'], '--dir', str(download)])
        if sha256_file(download / asset['name']) != asset['sha256']:
            raise GenerationError('GitHub_redownload_digest_mismatch')
    archive = next(a for a in package['assets'] if a['name'] == package['archive'])
    restored = attempt / 'restored'
    restoration = restore_archive(download / archive['name'], restored, archive['sha256'])
    for line in (download / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1); safe_relative(name)
        if sha256_file(download / name) != expected:
            raise GenerationError('checkpoint_sidecar_checksum_mismatch')
    copied_rejection = _verify_restored_public(restored)
    result = {'status': 'GITHUB_VERIFIED', 'release_id': release['id'], 'draft': release['draft'],
              'url': release['html_url'], 'target_commitish': release['target_commitish'],
              'assets': [{**a, 'asset_id': current[a['name']]['id']} for a in package['assets']],
              'restoration': restoration, 'canonical_plan_path_preserved_in_forensic_copy': True,
              'copied_plan_load_rejection': copied_rejection, 'target_commit_on_dedicated_branch': True,
              'research_api_requests_sent_by_upload': 0}
    write_json(attempt / 'verification.json', result)
    return result
