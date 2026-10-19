"""Small Goal015A router; saved-only commands never start or resume generation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import uuid


@contextmanager
def _completion_lock(root):
    """Serialize save progress so concurrent finish commands cannot create Drafts."""
    from ...experiments.goal015 import runtime as parent
    with parent.path_safe(root/'completion.lock').open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise parent.RuntimeErrorSafe('completion evaluation/save already active') from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def finish(plan_path, *, allow_create=False):
    """Complete CPU evaluation/save, reusing the same checkpoint on save retry."""
    from . import runtime
    from ...experiments.goal015 import runtime as parent
    plan = runtime.verify_plan(plan_path)
    root = Path(plan['canonical_root'])
    state = runtime.status(plan_path)
    parent.require(state.get('status') == 'COMPLETE' and state.get('calls_settled') == 8
                   and state.get('calls_not_sent') == 0 and state.get('calls_uncertain') == 0,
                   'child is partial or uncertain; finish cannot generate missing calls')
    with _completion_lock(root):
        return _finish_locked(plan_path, plan, allow_create=allow_create)


def _finish_locked(plan_path, plan, *, allow_create):
    from . import evaluate, checkpoint
    from ...experiments.goal015 import runtime as parent
    root = Path(plan['canonical_root'])
    ledger = parent.descriptor(root/'enabled/ledger.json')
    progress_path = root/'completion-progress.json'
    if progress_path.exists():
        progress = parent.read_json(progress_path)
        parent.require(progress.get('plan_id') == plan['plan_id'] and progress.get('child_ledger') == ledger,
                       'saved completion progress belongs to different child state')
    else:
        progress = {'schema': 'goal015a_completion_progress_v1', 'plan_id': plan['plan_id'],
                    'child_ledger': ledger, 'status': 'EVALUATION_PENDING'}
    if 'evaluation_summary' in progress:
        summary = evaluate.find_existing_evaluation(plan_path)
        parent.require(summary is not None and summary == parent.bound(progress['evaluation_summary'])
                       and summary['evaluation_artifact'] == progress['evaluation_artifact'],
                       'saved completion evaluation no longer matches authenticated current sources')
        parent.bound(progress['evaluation_artifact'])
    else:
        summary = evaluate.evaluate_plan(plan_path)
        evaluation = summary['evaluation_artifact']
        progress.update(evaluation_summary=parent.descriptor(Path(evaluation['path']).parent/'summary.json'),
                        evaluation_artifact=evaluation, status='EVALUATED')
        parent._save(progress_path, progress)
    if 'checkpoint' not in progress:
        saved = checkpoint.create_save_only(plan_path)
        progress.update(checkpoint=parent.descriptor(Path(saved['path'])/'checkpoint.json'),
                        checkpoint_directory=saved['path'], status='LOCAL_VERIFIED')
        parent._save(progress_path, progress)
    parent.bound(progress['checkpoint'])
    verification = root/'github-verifications'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid.uuid4().hex[:6])
    previous = progress.get('github', {})
    proof = checkpoint.upload_checkpoint(progress['checkpoint_directory'], verification,
        release_id=previous.get('release_id'), allow_create=allow_create)
    parent.require(proof.get('status') == 'GITHUB_VERIFIED', 'save verification incomplete')
    progress.update(status='GITHUB_VERIFIED', github=proof, verification_directory=str(verification))
    parent._save(progress_path, progress)
    return {'status': 'COMPLETE_SAVED', 'plan_id': plan['plan_id'], 'planned_calls': 80,
            'settled_attempts': 80, 'reconstruction_success_not_implied': True,
            'evaluated_protocol_valid_rows': summary['evaluated_protocol_valid_rows'],
            'actual_metric_rows': summary['actual_metric_rows'], 'github': proof,
            'completion_progress': parent.descriptor(progress_path), 'research_api_requests_sent_by_finish': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Goal015A: separately approved residual8, combined USD7.00')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preview', 'status', 'start', 'resume', 'run', 'finish', 'evaluate', 'checkpoint-local'):
        command = sub.add_parser(name)
        command.add_argument('--plan', type=Path, required=True)
        if name in ('start', 'resume', 'run'):
            command.add_argument('--pricing-evidence', type=Path, required=True)
        if name == 'evaluate':
            command.add_argument('--output-dir', type=Path)
        if name == 'finish':
            command.add_argument('--allow-create', action='store_true')
    command = sub.add_parser('checkpoint-upload')
    command.add_argument('--checkpoint-dir', type=Path, required=True)
    command.add_argument('--verification-dir', type=Path, required=True)
    command.add_argument('--release-id', type=int)
    command.add_argument('--allow-create', action='store_true')
    args = parser.parse_args(argv)
    exit_code = 0
    if args.command in ('preview', 'status', 'start', 'resume', 'run'):
        from . import runtime
        if args.command in ('preview', 'status'):
            result = getattr(runtime, args.command)(args.plan)
        else:
            result = runtime.execute(args.plan, pricing_evidence=args.pricing_evidence,
                                     resume=args.command == 'resume')
            complete = (result.get('status') == 'COMPLETE' and result.get('calls_settled') == 8
                        and result.get('calls_not_sent') == 0 and result.get('calls_uncertain') == 0)
            if not complete:
                exit_code = 2
            elif args.command == 'run':
                result = finish(args.plan, allow_create=True)
    elif args.command == 'finish':
        result = finish(args.plan, allow_create=args.allow_create)
    elif args.command == 'evaluate':
        from .evaluate import evaluate_plan
        result = evaluate_plan(args.plan, args.output_dir)
    elif args.command == 'checkpoint-local':
        from .checkpoint import create_save_only
        result = create_save_only(args.plan)
    else:
        from .checkpoint import upload_checkpoint
        result = upload_checkpoint(args.checkpoint_dir, args.verification_dir,
            release_id=args.release_id, allow_create=args.allow_create)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == '__main__':
    try:
        code = main()
    except KeyboardInterrupt:
        print(json.dumps({'status': 'INTERRUPTED', 'automatic_retry': False}))
        code = 130
    except Exception as error:
        # Only deliberate safe runtime messages are printable; suppress other details.
        from ...experiments.goal015.runtime import RuntimeErrorSafe
        print(json.dumps({'status': 'FAILED', 'error_type': type(error).__name__,
                          'reason': str(error) if isinstance(error, RuntimeErrorSafe) else 'inspect saved local evidence',
                          'automatic_retry': False, 'inspect_saved_state_before_next_action': True}))
        code = 1
    raise SystemExit(code)
