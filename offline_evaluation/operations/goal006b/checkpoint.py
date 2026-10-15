"""Local-only supplemental snapshots; original 781 responses remain references.

The caller holds original and supplemental run locks. This saver never acquires
those locks again, invokes generation, modifies a ledger, or accesses GitHub.
"""
from __future__ import annotations

from pathlib import Path
import re
import uuid

from offline_evaluation.continuation_v1.checkpoint import _archive, scan_public_file
from offline_evaluation.server_v2.checkpoint import restore_archive, scan_file
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json, utc_now
from . import planning, runner

CHECKPOINT_ROOT = Path('/root/minecraft-checkpoints')


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise GenerationError('supplement_checkpoint_regular_nonsymlink_file_required')
    return path


def _inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise GenerationError('supplement_checkpoint_symlink_rejected')
    files = set()
    for path in root.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise GenerationError('supplement_checkpoint_symlink_or_special_file_rejected')
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _source_controls(identity):
    snapshot = identity['source_snapshot']
    source = read_json(_regular(snapshot['plan_path']))
    source_directory = Path(source['identity']['enabled_directory'])
    pairs = [(snapshot['plan_path'], snapshot['plan_sha256']),
             (source_directory / 'ledger.json', snapshot['ledger_sha256']),
             (source_directory / 'approval.json', snapshot['approval_sha256']),
             (source['identity']['authority_path'], snapshot['authority_sha256'])]
    actual = {str(_regular(path)): sha256_file(path) for path, _ in pairs}
    if any(actual[str(Path(path).absolute())] != expected for path, expected in pairs):
        raise GenerationError('supplement_checkpoint_original_control_changed')
    return actual


def _entries(plan, directory, ledger):
    identity = plan['identity']
    canonical = _regular(identity['plan_path'])
    if (planning.load_plan(canonical) != plan or directory != Path(identity['enabled_directory'])
            or read_json(_regular(directory / 'ledger.json')) != ledger):
        raise GenerationError('supplement_checkpoint_current_canonical_ledger_required')
    if ledger.get('version') != 'goal006b_ledger_v1' or ledger.get('plan_identity_hash') != plan['identity_hash']:
        raise GenerationError('supplement_checkpoint_ledger_identity_mismatch')
    approval = read_json(_regular(directory / 'approval.json'))
    runner._approval(plan, approval, fresh=False)
    if read_json(_regular(directory / 'enabled.json')) != {
        'api_allowed': True, 'plan_identity_hash': plan['identity_hash'],
        'approval_hash': object_hash(approval), 'enabled_directory': str(directory)}:
        raise GenerationError('supplement_checkpoint_enabled_mismatch')
    authority = _regular(identity['authority_path'])
    if read_json(authority) != runner._authority(plan, approval):
        raise GenerationError('supplement_checkpoint_authority_mismatch')
    entries = [('supplement/plan.json', canonical), ('supplement/authority.json', authority)]
    allowed = set()
    for name in ('approval.json', 'enabled.json', 'ledger.json'):
        allowed.add(name)
        entries.append(('supplement/enabled/' + name, _regular(directory / name)))
    for name in ('run.lock', 'summary.json'):
        if (directory / name).exists():
            allowed.add(name)
            entries.append(('supplement/enabled/' + name, _regular(directory / name)))
    raw_scopes = {}
    seen = set()
    rows = ledger.get('entries')
    if not isinstance(rows, list) or len(rows) > len(identity['schedule']):
        raise GenerationError('supplement_checkpoint_invalid_entry_count')
    for index, entry in enumerate(rows):
        record, intent = identity['schedule'][index], identity['intents'][index]
        attempt_id = entry.get('attempt_id', '')
        if (not re.fullmatch(r'[0-9a-f]{32}', attempt_id) or attempt_id in seen
                or entry.get('local_index') != index or entry.get('schedule_index') != record['schedule_index']
                or entry.get('provider') != record['provider'] or entry.get('input_id') != record['input_id']
                or entry.get('intent') != intent['kind'] or entry.get('source_attempt_id') != intent['source_attempt_id']
                or entry.get('retry') is not (intent['kind'] == 'manual_auth_retry')
                or entry.get('status') not in {'reserved', 'success', 'failed'}):
            raise GenerationError('supplement_checkpoint_attempt_scope_mismatch')
        if index < len(rows) - 1 and (entry['status'] == 'reserved' or entry.get('usage') is None):
            raise GenerationError('supplement_checkpoint_continued_after_unknown_usage')
        seen.add(attempt_id)
        claim_relative = 'reservations/' + attempt_id + '.json'
        claim = _regular(directory / claim_relative)
        if read_json(claim) != runner._claim(plan, entry):
            raise GenerationError('supplement_checkpoint_claim_mismatch')
        allowed.add(claim_relative)
        entries.append(('supplement/enabled/' + claim_relative, claim))
        for name, descriptor in entry['artifacts'].items():
            relative = f'attempts/{attempt_id}/{name}'
            if name not in {'raw.json', 'parsed.json', 'voxels.json'} or descriptor['path'] != relative:
                raise GenerationError('supplement_checkpoint_artifact_path_mismatch')
            path = _regular(directory / relative)
            if sha256_file(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
                raise GenerationError('supplement_checkpoint_artifact_hash_mismatch')
            allowed.add(relative)
            entries.append(('supplement/enabled/' + relative, path))
            if name == 'raw.json':
                raw_scopes[str(path)] = {'provider': entry['provider'], 'sha256': descriptor['sha256'],
                    'accepted_models': identity['pricing'][entry['provider']]['accepted_response_model_ids']}
    if _inventory(directory) != allowed:
        raise GenerationError('supplement_checkpoint_unexpected_or_missing_enabled_file')
    expected_dirs = {Path(relative).parent.as_posix() for relative in allowed}
    expected_dirs |= {'attempts', 'reservations'}
    for path in directory.rglob('*'):
        if path.is_dir() and path.relative_to(directory).as_posix() not in expected_dirs:
            raise GenerationError('supplement_checkpoint_unexpected_enabled_directory')
    revisions = canonical.parent / 'pricing-revisions'
    revision_names = set()
    if revisions.exists():
        for index, path in enumerate(sorted(revisions.iterdir()), 1):
            if path.name != f'{index:04d}.json':
                raise GenerationError('supplement_checkpoint_pricing_revision_sequence_mismatch')
            revision_names.add('pricing-revisions/' + path.name)
            entries.append(('supplement/pricing-revisions/' + path.name, _regular(path)))
    expected_tree = {'plan.json'} | revision_names | {'enabled/' + name for name in allowed}
    if _inventory(canonical.parent) != expected_tree:
        raise GenerationError('supplement_checkpoint_unexpected_plan_tree_file')
    for path in canonical.parent.iterdir():
        if path.name not in {'plan.json', 'enabled', 'pricing-revisions'}:
            raise GenerationError('supplement_checkpoint_unexpected_plan_tree_entry')
    return entries, raw_scopes


def save_local(plan, directory, ledger, reason):
    """Save a new verified local archive while caller holds all live-run locks."""
    directory = Path(directory).absolute()
    entries, raw_scopes = _entries(plan, directory, ledger)
    identity = plan['identity']
    source_before = _source_controls(identity)
    before = {str(path): sha256_file(path) for _, path in entries}
    stamp = utc_now().replace('-', '').replace(':', '').replace('.', '') + '-' + uuid.uuid4().hex[:8]
    checkpoint_id = f"goal006b-{plan['identity_hash'][:12]}-{len(ledger['entries']):04d}-{stamp}"
    destination = planning.safe_path(CHECKPOINT_ROOT / checkpoint_id)
    destination.mkdir(parents=True, exist_ok=False)
    execution = runner.summary(plan, ledger)
    source_reference = {'schema': 'goal006b_preserved_source_reference_v1',
        'repository': 'catlover-bot/Minecraft-view-dataset',
        'original_checkpoint': identity['checkpoint_reference'],
        'source_plan_identity_hash': identity['source_snapshot']['plan_identity_hash'],
        'source_controls_sha256': source_before,
        'original_attempts': identity['source_snapshot']['attempted_inputs'],
        'original_halt_reason': identity['source_snapshot']['halt_reason'],
        'original_unknown_usage_reservation_reconciled': False,
        'scope': 'Original 781 attempts and earlier sources remain immutable referenced assets; no response bytes duplicated.'}
    write_json(destination / 'SOURCE_REFERENCE.json', source_reference)
    entries.append(('SOURCE_REFERENCE.json', destination / 'SOURCE_REFERENCE.json'))
    metadata = {'schema': 'goal006b_local_checkpoint_v1', 'checkpoint_id': checkpoint_id,
        'created_at_utc': utc_now(), 'reason': reason, 'plan_identity_hash': plan['identity_hash'],
        'canonical_plan_sha256': sha256_file(identity['plan_path']), 'code_commit': identity['commit'],
        'execution': execution, 'request_active': False,
        'terminal_reserved_attempts': sum(e['status'] == 'reserved' for e in ledger['entries']),
        'supplemental_unknown_usage_attempts': sum(e.get('usage') is None for e in ledger['entries']),
        'reserved_or_unknown_attempts_preserved_without_reconciliation': True,
        'github_status': 'NOT_UPLOADED', 'research_api_requests_sent_by_checkpoint': 0}
    archive = destination / 'supplement-point-in-time.tar.gz'
    manifest = _archive(archive, entries, metadata, raw_scopes)
    after = {path: sha256_file(path) for path in before}
    if before != after or source_before != _source_controls(identity):
        raise GenerationError('supplement_checkpoint_originals_changed_during_archive')
    restored = destination / 'restored-verification'
    restoration = restore_archive(archive, restored, sha256_file(archive))
    if read_json(restored / 'FILES.json') != manifest:
        raise GenerationError('supplement_checkpoint_restored_manifest_mismatch')
    scopes_by_name = {name: raw_scopes.get(str(path)) for name, path in entries}
    for row in manifest['files']:
        scan_public_file(restored / row['path'], scopes_by_name[row['path']])
    scan_file(restored / 'FILES.json')
    copied_path = restored / 'supplement/plan.json'
    try:
        planning.load_plan(copied_path)
    except GenerationError as exc:
        if 'canonical' not in str(exc):
            raise GenerationError('supplement_checkpoint_copy_rejected_for_unexpected_reason') from None
        copy_rejection = str(exc)
    else:
        raise GenerationError('supplement_checkpoint_copied_plan_accepted')
    # Restore checks every file, while these explicit comparisons make the
    # preserved approval, authority, claim ledger and referenced source visible.
    if (read_json(restored / 'supplement/enabled/ledger.json') != ledger
            or read_json(restored / 'SOURCE_REFERENCE.json') != source_reference):
        raise GenerationError('supplement_checkpoint_restored_control_mismatch')
    write_json(destination / 'snapshot.json', {**metadata, 'source_reference': source_reference,
        'original_sha256': before, 'originals_unchanged': True,
        'archive_sha256': sha256_file(archive), 'file_count': len(manifest['files']),
        'restoration': restoration, 'copied_plan_load_rejection': copy_rejection})
    (destination / 'RESTORE.md').write_text(
        'This is a local-only supplemental checkpoint; GitHub status is NOT_UPLOADED. '
        'Verify SHA256SUMS and use offline_evaluation.server_v2.checkpoint.restore_archive '
        'with the recorded SHA256 into a NEW directory. All original 781 responses and '
        'prior sources remain referenced by SOURCE_REFERENCE.json; do not recreate them. '
        'The copied plan is forensic and cannot activate. Reserved/unknown attempts '
        'remain reservations and must never be cleared or automatically retried.\n')
    names = [archive.name, 'FILES.json', 'snapshot.json', 'SOURCE_REFERENCE.json', 'RESTORE.md']
    for name in names[1:]:
        scan_file(destination / name)
    (destination / 'SHA256SUMS').write_text(''.join(f'{sha256_file(destination / name)}  {name}\n' for name in names))
    names.append('SHA256SUMS')
    result = {'version': 'goal006b_local_checkpoint_package_v1', 'status': 'LOCAL_VERIFIED',
        'github_status': 'NOT_UPLOADED', 'checkpoint_directory': str(destination),
        'checkpoint_id': checkpoint_id, 'plan_identity_hash': plan['identity_hash'],
        'archive': archive.name, 'archive_sha256': sha256_file(archive),
        'assets': [{'name': name, 'sha256': sha256_file(destination / name),
                    'bytes': (destination / name).stat().st_size} for name in names],
        'restoration': restoration, 'copied_plan_load_rejection': copy_rejection,
        'execution': execution, 'originals_unchanged': True,
        'source_responses_duplicated': False, 'research_api_requests_sent_by_checkpoint': 0}
    write_json(destination / 'checkpoint.json', result)
    scan_file(destination / 'checkpoint.json')
    if before != {path: sha256_file(path) for path in before} or source_before != _source_controls(identity):
        raise GenerationError('supplement_checkpoint_originals_changed_during_restore')
    return result
