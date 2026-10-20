"""Explicit Goal015C start/resume and generation-free evaluate/save commands."""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import uuid

from . import runtime


@contextmanager
def completion_lock(plan):
    root = runtime.path_safe(plan['canonical_root'])
    # No disabled plan can create completion state.
    with runtime.locks(plan):
        runtime.verify_enabled(plan)
    with runtime.path_safe(root/'completion.lock').open('a+b') as stream:
        runtime.files._fsync_directory(root)
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise runtime.RuntimeErrorSafe('finish or save already active') from None
        try:
            # Recheck after obtaining completion lock; running generation holds
            # execution.lock, and resume must obtain completion.lock too.
            with runtime.locks(plan):
                runtime.verify_enabled(plan)
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def _fresh_verification(root):
    return root/'github-verification'/('attempt-'+uuid.uuid4().hex)


def _checkpoint_from_progress(plan,progress):
    value = progress.get('checkpoint')
    if value is None:
        return None
    runtime.require(runtime.descriptor(value['checkpoint']['path']) == value['checkpoint'],
                    'saved checkpoint receipt changed')
    directory = runtime.path_safe(value['path'])
    runtime.require(directory.is_relative_to(Path(plan['canonical_root'])/'checkpoints'),
                    'checkpoint escaped canonical diagnostic root')
    from . import checkpoint
    package = runtime.read_json(directory/'checkpoint.json')
    runtime.require(package['plan_id'] == plan['plan_id'],'checkpoint belongs to another plan')
    checkpoint.verify_archive(directory/'goal015c-state.tar.gz',package['archive_sha256'])
    return value


def finish(plan_path, *, release_id=None, allow_create=False, github=None):
    """Evaluate current immutable observations and save; never call execute."""
    from . import evaluate, checkpoint
    plan = runtime.verify_plan(plan_path)
    runtime.require(github is None or plan.get('fixture_only') is True,'fake GitHub forbidden for production')
    root = Path(plan['canonical_root']); progress_path = root/'completion-progress.json'
    with completion_lock(plan):
        current = runtime.descriptor(root/'enabled/ledger.json')
        progress = runtime.read_json(progress_path) if progress_path.exists() else {
            'schema':'goal015c_completion_progress_v1','plan_id':plan['plan_id'],'ledger':current}
        runtime.require(progress['plan_id'] == plan['plan_id'],'completion belongs to another plan')
        if progress['ledger'] != current:
            # A later explicit resume can legitimately add NOT_SENT outcomes.
            # Keep the previous receipt and start a new generation-free save.
            archive = root/('completion-progress-prior-'+uuid.uuid4().hex+'.json')
            runtime._save(archive,progress,exclusive=True)
            progress = {'schema':'goal015c_completion_progress_v1','plan_id':plan['plan_id'],'ledger':current}
        try:
            summary = evaluate.find_existing_evaluation(plan_path)
            if summary is None:
                summary = evaluate.evaluate_plan(plan_path)
            runtime.require(runtime.descriptor(root/'enabled/ledger.json') == current,
                            'ledger changed during evaluation')
            progress['evaluation'] = summary['evaluation_artifact']
            progress['evaluation_summary'] = summary['summary_artifact']
            progress['status'] = 'EVALUATED'; runtime._save(progress_path,progress)
            saved = _checkpoint_from_progress(plan,progress)
            if saved is None:
                saved = checkpoint.create_save_only(plan_path,reason='EVALUATED_FINISH')
                progress['checkpoint'] = saved
                progress['status'] = 'LOCAL_VERIFIED'; runtime._save(progress_path,progress)
            existing = progress.get('github_verification')
            if existing:
                runtime.require(runtime.descriptor(existing['path']) == existing,'GitHub verification receipt changed')
                proof = runtime.read_json(existing['path'])
            else:
                verification_dir = _fresh_verification(root)
                proof = checkpoint.upload_checkpoint(saved['path'],verification_dir,
                    release_id=release_id,allow_create=allow_create,github=github)
                progress['github_verification'] = runtime.descriptor(verification_dir/'verification.json')
            runtime.require(proof.get('status') == 'GITHUB_VERIFIED'
                and proof.get('archive_sha256') == saved['archive_sha256']
                and proof.get('draft') is True
                and proof.get('restoration',{}).get('plan_id') == plan['plan_id'],
                'remote checkpoint verification incomplete or mismatched')
            state = runtime.status(plan_path)
            complete = state['calls_settled'] == 12 and state['calls_uncertain'] == 0 and state['calls_not_sent'] == 0
            progress['status'] = 'COMPLETE_SAVED' if complete else 'PARTIAL_SAVED'
            progress.pop('failure',None); runtime._save(progress_path,progress)
            return {**state,'status':progress['status'],'evaluation':summary,
                    'checkpoint':saved,'github_verification':progress['github_verification'],
                    'research_api_requests_sent_by_finish':0,'generation_reexecuted':False}
        except Exception as exc:
            progress['status'] = 'SAVE_FAILED'
            progress['failure'] = {'type':type(exc).__name__,
                'reason':str(exc) if isinstance(exc,runtime.RuntimeErrorSafe) else 'finish failed; inspect saved checkpoint and retry finish/save-only'}
            runtime._save(progress_path,progress)
            return {'status':'SAVE_FAILED','plan_id':plan['plan_id'],'checkpoint':progress.get('checkpoint'),
                    'failure':progress['failure'],'research_api_requests_sent_by_finish':0,
                    'generation_reexecuted':False,'recovery':'Retry finish or save-only; do not start or resume generation for a save failure.'}


def save_only(plan_path, *, checkpoint_dir=None, verification_dir=None, release_id=None,
              allow_create=False, github=None):
    """Upload/reverify the same local package without any generation/evaluation."""
    from . import checkpoint
    plan = runtime.verify_plan(plan_path); root = Path(plan['canonical_root'])
    runtime.require(github is None or plan.get('fixture_only') is True,'fake GitHub forbidden for production')
    with completion_lock(plan):
        if checkpoint_dir is None:
            progress_path = root/'completion-progress.json'
            progress = runtime.read_json(progress_path) if progress_path.exists() else {}
            saved = _checkpoint_from_progress(plan,progress)
            if saved is None:
                saved = checkpoint.create_save_only(plan_path)
            directory = Path(saved['path'])
        else:
            directory = runtime.path_safe(Path(checkpoint_dir).absolute())
            runtime.require(directory.is_relative_to(root/'checkpoints'),'checkpoint escaped canonical diagnostic root')
            package = runtime.read_json(directory/'checkpoint.json')
            runtime.require(package['plan_id'] == plan['plan_id'],'checkpoint belongs to another plan')
        target = runtime.path_safe(Path(verification_dir).absolute()) if verification_dir else _fresh_verification(root)
        runtime.require(target.is_relative_to(root/'github-verification'),'verification escaped canonical diagnostic root')
        proof = checkpoint.upload_checkpoint(directory,target,release_id=release_id,allow_create=allow_create,github=github)
        return {'status':'SAVED_ONLY','plan_id':plan['plan_id'],'checkpoint_dir':str(directory),
                'github_verification':runtime.descriptor(target/'verification.json'),'verification':proof,
                'research_api_requests_sent':0,'generation_reexecuted':False}


def parser():
    out = argparse.ArgumentParser(description=__doc__)
    commands = out.add_subparsers(dest='command',required=True)
    for name in ('preview','status','preflight','start','resume','run','finish','save-only','evaluate','checkpoint-local'):
        item = commands.add_parser(name); item.add_argument('--plan',required=True,type=Path)
        if name in ('preflight','start','resume','run'):
            item.add_argument('--pricing-evidence',type=Path)
        if name == 'preflight':
            item.add_argument('--require-unstarted',action='store_true')
        if name in ('run','finish','save-only'):
            item.add_argument('--allow-create',action='store_true')
            item.add_argument('--release-id',type=int)
        if name == 'save-only':
            item.add_argument('--checkpoint-dir',type=Path)
            item.add_argument('--verification-dir',type=Path)
    return out


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        command = args.command
        if command == 'preview':
            result = runtime.preview(args.plan)
        elif command == 'status':
            result = runtime.status(args.plan)
        elif command == 'preflight':
            result = runtime.preflight(args.plan,pricing_evidence=args.pricing_evidence,
                                       require_unstarted=args.require_unstarted)
        elif command in ('start','resume','run'):
            result = runtime.execute(args.plan,pricing_evidence=args.pricing_evidence,resume=command=='resume')
            if command == 'run' and result['status'] == 'COMPLETE' and result['calls_settled'] == 12 and result['calls_uncertain'] == 0:
                result = finish(args.plan,release_id=args.release_id,allow_create=args.allow_create)
        elif command == 'finish':
            result = finish(args.plan,release_id=args.release_id,allow_create=args.allow_create)
        elif command == 'save-only':
            result = save_only(args.plan,checkpoint_dir=args.checkpoint_dir,verification_dir=args.verification_dir,
                               release_id=args.release_id,allow_create=args.allow_create)
        elif command == 'evaluate':
            from .evaluate import evaluate_plan
            result = evaluate_plan(args.plan)
        else:
            from .checkpoint import create_save_only
            result = create_save_only(args.plan)
        print(json.dumps(result,sort_keys=True),flush=True)
        return 2 if result.get('status') in ('SAFE_STOP','PARTIAL_WITH_UNCERTAIN_CALLS','PARTIAL_SAVED','SAVE_FAILED') else 0
    except Exception as exc:
        print(json.dumps({'status':'REFUSED','exception_type':type(exc).__name__,
            'reason':str(exc) if isinstance(exc,runtime.RuntimeErrorSafe) else 'validation or operation failed; inspect local evidence',
            'automatic_retry':False,'generation_reexecuted_for_save':False},sort_keys=True),flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
