"""Verified checkpoint recovery and same-ledger continuation operations.

This module is deliberately outside every frozen implementation-hash directory.
It never enables a plan. Paid work requires the explicit --resume invocation and
an existing, validated human approval; ordinary invocation is read-only preview.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from offline_evaluation.continuation_v1 import plan as planning, runner
from offline_evaluation.server_v2.data import sha256_file
from offline_evaluation.server_v2.generation import GenerationError, read_json, utc_now
from offline_evaluation.server_v2.io_utils import ROOT

CANONICAL_PLAN = Path('/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json')
ANCHOR = Path('/root/minecraft-checkpoints/goal006-db5533bffb58-0320-20261014T150741439937Z-ad2c6d7f')
ANCHOR_RELEASE = 412856648
ANCHOR_ARCHIVE_ASSET = 640740758
ANCHOR_ARCHIVE_SHA256 = 'bb643bcd59872fbc4d869b0ba5628e585d54a4cc5827794aa8bfc42eba98b9cc'
CHECKPOINT_ROOT = Path('/root/minecraft-checkpoints')


def _path(value):
    candidate = Path(value).absolute()
    if '..' in candidate.parts or any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise GenerationError('operations_symlink_or_traversal_rejected')
    return candidate


def _stamp():
    return utc_now().replace('-', '').replace(':', '').replace('.', '') + '-' + uuid.uuid4().hex[:8]


def resume_command(plan_path=CANONICAL_PLAN):
    return shlex.join([str(ROOT / '.venv-server-v2-20261013/bin/python'), '-B',
                      str(Path(__file__).resolve()), '--plan', str(plan_path), '--resume'])


def recovery_command(checkpoint_directory, *, release_id=None, allow_create=False):
    command = [str(ROOT / '.venv-server-v2-20261013/bin/python'), '-B',
               str(Path(__file__).resolve()), '--verify-checkpoint', str(checkpoint_directory)]
    if release_id is not None:
        command += ['--release-id', str(release_id)]
    if allow_create:
        command.append('--allow-create')
    return shlex.join(command)


@contextmanager
def _operations_lock(root, identity_hash):
    """Serializes this operations entry; frozen runner retains its own locks."""
    root = _path(root)
    root.mkdir(parents=True, exist_ok=True)
    with _path(root / ('.resume-' + identity_hash + '.lock')).open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GenerationError('goal006a_operations_already_active') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _controls(plan):
    identity = plan['identity']
    directory = _path(identity['enabled_directory'])
    paths = [_path(identity['plan_path']), _path(identity['authority_path'])]
    paths += [_path(directory / name) for name in ('approval.json', 'enabled.json', 'ledger.json')]
    if any(not path.is_file() for path in paths):
        raise GenerationError('existing_approved_canonical_run_required')
    return {str(path): sha256_file(path) for path in paths}


def _checkpoint_binding(directory, plan, *, ledger_hash=None):
    directory = _path(directory)
    package = read_json(_path(directory / 'checkpoint.json'))
    snapshot = read_json(_path(directory / 'snapshot.json'))
    ledger_path = str(_path(Path(plan['identity']['enabled_directory']) / 'ledger.json'))
    if (package.get('plan_identity_hash') != plan['identity_hash']
            or snapshot.get('plan_identity_hash') != plan['identity_hash']
            or snapshot.get('canonical_plan_sha256') != sha256_file(plan['identity']['plan_path'])):
        raise GenerationError('checkpoint_does_not_bind_original_canonical_plan')
    saved_hash = snapshot.get('original_sha256', {}).get(ledger_path)
    if not saved_hash or (ledger_hash is not None and saved_hash != ledger_hash):
        raise GenerationError('checkpoint_not_current_ledger')
    return package, snapshot, saved_hash


def matching_checkpoint(plan, checkpoint_root=CHECKPOINT_ROOT):
    """Select an immutable saved package for the exact current ledger bytes."""
    ledger_hash = sha256_file(Path(plan['identity']['enabled_directory']) / 'ledger.json')
    root = _path(checkpoint_root)
    candidates = []
    for directory in sorted(root.glob('goal006-' + plan['identity_hash'][:12] + '-*')):
        if directory.is_symlink() or not directory.is_dir():
            continue
        if not (directory / 'checkpoint.json').is_file() or not (directory / 'snapshot.json').is_file():
            continue  # Incomplete local packages cannot authorize a resume.
        package, snapshot, saved_hash = _checkpoint_binding(directory, plan)
        if saved_hash == ledger_hash:
            candidates.append(directory)
    if not candidates:
        raise GenerationError('current_ledger_checkpoint_missing; checkpoint_recovery_required_before_resume')
    # Multiple snapshots of identical ledger bytes are equivalent; newest name
    # wins, and it still must pass full numeric-ID GitHub verification.
    return candidates[-1]


def _verified_binding(report, plan, expected_ledger_hash, expected_count):
    if (report.get('status') != 'GITHUB_VERIFIED'
            or report.get('plan_identity_hash') != plan['identity_hash']
            or report.get('verifiedledger_sha256') != expected_ledger_hash
            or report.get('attempted_inputs') != expected_count):
        raise GenerationError('recovery_report_does_not_match_expected_ledger')


def preview(plan_path=CANONICAL_PLAN, *, checkpoint_root=CHECKPOINT_ROOT):
    """Only local reads: no approval, reservations, network or new files."""
    plan = planning.load_plan(plan_path)
    status = runner.status(plan_path)
    current = None
    if status['approved']:
        try:
            current = str(matching_checkpoint(plan, checkpoint_root))
        except GenerationError:
            pass
    try:
        planning.current_pricing(plan['identity'], fresh=True)
        fresh = True
    except GenerationError:
        fresh = False
    return {**status, 'api_allowed': False, 'preview_only': True,
            'pricing_current_utc_day': fresh,
            'current_ledger_checkpoint_candidate': current,
            'github_verification_required_before_resume': True,
            'resume_command': resume_command(plan_path),
            'research_api_requests_sent_by_preview': 0}


def _save_callback(verification_root):
    def callback(plan, directory, ledger, reason, *, github=False):
        from offline_evaluation.continuation_v1.aggregate import evaluate_verified
        from offline_evaluation.continuation_v1.checkpoint import safe_checkpoint
        from offline_evaluation.operations.goal006a import archive
        local = None
        stage = 'incremental_evaluation'
        try:
            evaluate_verified(plan, directory, ledger)
            stage = 'local_checkpoint'
            local = safe_checkpoint(plan, directory, ledger, reason, github=False)
            if local.get('status') != 'LOCAL_VERIFIED':
                raise GenerationError('local_checkpoint_not_verified')
            stage = 'github_checkpoint'
            result = archive.recover(local['checkpoint_directory'],
                                     _path(verification_root) / ('segment-' + _stamp()),
                                     allow_create=True)
            _verified_binding(result, plan, sha256_file(directory / 'ledger.json'), len(ledger['entries']))
            return {'status': 'GITHUB_VERIFIED', 'checkpoint_directory': local['checkpoint_directory'],
                    'github': result, 'reason': reason}
        except Exception as exc:
            # Provider bodies, auth headers, command stderr and URLs are never
            # serialized here. Archive errors carry a constrained safe report.
            safe = {'stage': stage, 'code': type(exc).__name__, 'returncode': None, 'http_status': None}
            if isinstance(exc, archive.RecoveryError):
                safe = exc.report
            result = {'status': 'GITHUB_SAVE_FAILED', 'reason': reason, 'error': safe,
                      'generation_ledger_unchanged_by_save_failure': True}
            if local and local.get('checkpoint_directory'):
                result['checkpoint_directory'] = local['checkpoint_directory']
                result['recovery_command'] = recovery_command(local['checkpoint_directory'], allow_create=True)
            return result
    return callback


def resume(plan_path=CANONICAL_PLAN, *, checkpoint_root=CHECKPOINT_ROOT,
           verification_root=None, anchor_checkpoint=ANCHOR,
           anchor_release_id=ANCHOR_RELEASE, anchor_archive_asset_id=ANCHOR_ARCHIVE_ASSET,
           anchor_archive_sha256=ANCHOR_ARCHIVE_SHA256,
           transports=None):
    """Explicit paid entry, never called by preview or checkpoint-only recovery."""
    from offline_evaluation.operations.goal006a import archive
    plan = planning.load_plan(plan_path)
    verification_root = _path(verification_root or (Path(checkpoint_root) / 'goal006a-operations'))
    with _operations_lock(verification_root, plan['identity_hash']):
        status = runner.status(plan_path)
        if not status['approved']:
            raise GenerationError('original_human_approval_required; this_helper_never_activates')
        if status['halt_reason'] or status['used']['new']['unknown_usage_attempts']:
            raise GenerationError('existing_halt_or_unknown_usage; never_clear_or_resend')
        controls = _controls(plan)
        ledger_path = str(Path(plan['identity']['enabled_directory']) / 'ledger.json')
        anchor_package, anchor_snapshot, anchor_hash = _checkpoint_binding(anchor_checkpoint, plan)
        archive_asset = next((a for a in anchor_package['assets'] if a['name'] == anchor_package['archive']), None)
        if not archive_asset or archive_asset['sha256'] != anchor_archive_sha256:
            raise GenerationError('original_recovery_anchor_archive_mismatch')
        anchor_count = anchor_snapshot['execution']['attempted_inputs']
        if anchor_count > status['attempted_inputs']:
            raise GenerationError('current_ledger_smaller_than_recovery_anchor; never_recreate')
        anchor_report = archive.recover(anchor_checkpoint, verification_root / ('anchor-' + _stamp()),
                                        release_id=anchor_release_id, allow_create=False)
        _verified_binding(anchor_report, plan, anchor_hash, anchor_count)
        verified_archive = next((a for a in anchor_report.get('assets', [])
                                 if a['name'] == anchor_package['archive']), None)
        if not verified_archive or verified_archive.get('asset_id') != anchor_archive_asset_id:
            raise GenerationError('original_recovery_anchor_asset_id_mismatch')
        current = matching_checkpoint(plan, checkpoint_root)
        if current == _path(anchor_checkpoint) and controls[ledger_path] == anchor_hash:
            verified = anchor_report
        else:
            verified = archive.recover(current, verification_root / ('current-' + _stamp()), allow_create=False)
        _verified_binding(verified, plan, controls[ledger_path], status['attempted_inputs'])
        planning.current_pricing(plan['identity'], fresh=True)
        if controls != _controls(plan):
            raise GenerationError('canonical_controls_changed_during_recovery; retry_readonly_verification')
        # This operations lock serializes these wrappers. The frozen runner
        # independently acquires its normal nonblocking locks. No claim is made
        # of atomicity against an arbitrary competing legacy launch in the
        # interval between this final hash check and those lock acquisitions.
        result = runner.run(plan_path, permission=True, transports=transports,
                            checkpoint_callback=_save_callback(verification_root), github_checkpoints=True)
        return {**result, 'resume_command': resume_command(plan_path),
                'pre_resume_github_verification': verified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(CANONICAL_PLAN))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--verify-checkpoint')
    parser.add_argument('--release-id', type=int)
    parser.add_argument('--allow-create', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.verify_checkpoint:
            from offline_evaluation.operations.goal006a import archive
            result = archive.recover(args.verify_checkpoint,
                CHECKPOINT_ROOT / 'goal006a-operations' / ('manual-' + _stamp()),
                release_id=args.release_id, allow_create=args.allow_create)
        else:
            if args.release_id is not None or args.allow_create:
                parser.error('--release-id and --allow-create require --verify-checkpoint')
            if _path(args.plan) != CANONICAL_PLAN:
                raise GenerationError('operations_entry_requires_original_canonical_plan')
            result = resume(args.plan) if args.resume else preview(args.plan)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
        return 6 if result.get('operational_stop') or result.get('halt_reason') else 0
    except Exception as exc:
        from offline_evaluation.operations.goal006a import archive
        safe = {'stage': 'resume_preflight', 'code': type(exc).__name__, 'returncode': None, 'http_status': None}
        if isinstance(exc, archive.RecoveryError):
            safe = exc.report
        elif isinstance(exc, GenerationError):
            # Every GenerationError from this module/its frozen validators is
            # a fixed diagnostic code, never raw subprocess/provider output.
            safe['code'] = str(exc)
        print(json.dumps({'status': 'STOPPED', 'error': safe}, sort_keys=True), file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
