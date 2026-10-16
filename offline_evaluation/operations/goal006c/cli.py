"""Preview is the default. Paid suffix17 execution requires new exact real-TTY consent."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys
import uuid

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from offline_evaluation.operations.goal006c import planning, runner, archive
from offline_evaluation.operations.goal006a.archive import RecoveryError as OriginalRecoveryError
from offline_evaluation.server_v2.generation import GenerationError, read_json, utc_now
from offline_evaluation.server_v2 import providers


def confirmation_text(plan):
    return ('APPROVE GOAL006C ' + plan['identity_hash']
            + ' 17 NEW ONLY 943-959; NO 520 RETRY; MAX 991 CALLS; USD30; KEEP BOTH UNKNOWN HOLDS')


def tty_approval(plan):
    message = ('Goal006Cは未試行global 943–959の17件だけを送信します。940・941・942は送信しません。\n'
        '元401と補完520のhalt・usage不明予約を両方保持します。予約額は実請求額ではありません。\n'
        '累積上限991回・USD30、自動retry0・新規手動retry0。料金が発生し得ます。\n'
        '承認する場合だけ次の全文を実TTYで入力してください:\n' + confirmation_text(plan) + '\n> ')
    try:
        with ExitStack() as handles:
            try:
                reader = handles.enter_context(open('/dev/tty', 'r', encoding='utf-8', newline=''))
                writer = handles.enter_context(open('/dev/tty', 'w', encoding='utf-8', buffering=1))
            except OSError:
                raise GenerationError('goal006c_tty_unavailable_no_activation') from None
            if not reader.isatty() or not writer.isatty():
                raise GenerationError('goal006c_real_tty_required_no_activation')
            writer.write(message); writer.flush()
            answer = reader.readline()
    except (OSError, UnicodeError):
        raise GenerationError('goal006c_tty_io_error_no_activation') from None
    except EOFError:
        raise GenerationError('goal006c_approval_eof_no_activation') from None
    except KeyboardInterrupt:
        raise GenerationError('goal006c_approval_interrupted_no_activation') from None
    if answer == '':
        raise GenerationError('goal006c_approval_eof_no_activation')
    if answer != confirmation_text(plan) + '\n':
        raise GenerationError('goal006c_exact_approval_not_given_no_activation')
    return {'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
        'enabled_directory': plan['identity']['enabled_directory'], 'approved_at_utc': utc_now(),
        'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE, 'acknowledges_both_unknown_holds': True}


def save_checkpoint_only(directory, verification_directory, *, release_id=None, allow_create=False):
    """Use read-only locks; upload/recovery cannot call the generation runner."""
    directory = planning.safe_path(directory)
    package = archive._package(directory)
    snapshot = read_json(directory / 'snapshot.json')
    canonical = [Path(p) for p in snapshot['original_sha256'] if Path(p).name == 'plan.json']
    if len(canonical) != 1:
        raise GenerationError('goal006c_save_only_single_canonical_plan_required')
    source_planning, source_runner = archive._modules(package['package_kind'])
    plan = source_planning.load_plan(planning.safe_path(canonical[0])); identity = plan['identity']
    if plan['identity_hash'] != package['plan_identity_hash']:
        raise GenerationError('goal006c_save_only_checkpoint_identity_mismatch')
    locks = list(dict.fromkeys([*identity['source_snapshot']['lock_paths'],
        str(Path(identity['enabled_directory']) / 'run.lock'), identity['authority_path'] + '.lock']))
    with ExitStack() as handles:
        for name in locks:
            stream = handles.enter_context(planning.safe_path(name).open('rb'))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError('goal006c_save_only_generation_active') from None
        source_planning.load_plan(canonical[0]); source_runner.load_enabled(plan)
        return archive.recover(directory, verification_directory, release_id=release_id, allow_create=allow_create)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', metavar='NEW_DIRECTORY')
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--refresh-pricing', metavar='OFFICIAL_PRICING_JSON')
    mode.add_argument('--checkpoint', action='store_true', help='Save current enabled C run and verify a Draft; never generate')
    mode.add_argument('--save-checkpoint', metavar='EXISTING_PACKAGE', help='Upload/verify B or C package only; never generate')
    parser.add_argument('--source-plan', default=str(planning.SOURCE_PLAN))
    parser.add_argument('--checkpoint-verification')
    parser.add_argument('--pricing')
    parser.add_argument('--commit')
    parser.add_argument('--verification-directory')
    parser.add_argument('--release-id', type=int)
    parser.add_argument('--allow-create', action='store_true', help='Authorize Draft creation/missing asset upload in save-only mode')
    args = parser.parse_args(argv)
    try:
        if (args.allow_create or args.release_id or args.verification_directory) and not args.save_checkpoint:
            parser.error('save-only options require --save-checkpoint')
        if args.save_checkpoint:
            if args.plan:
                parser.error('--save-checkpoint must not include --plan')
            destination = args.verification_directory or str(Path('/root/minecraft-checkpoints/goal006c-operations') / ('save-' + uuid.uuid4().hex))
            result = save_checkpoint_only(args.save_checkpoint, destination,
                                         release_id=args.release_id, allow_create=args.allow_create)
        elif args.prepare:
            if args.plan or not all((args.checkpoint_verification, args.pricing, args.commit)):
                parser.error('--prepare requires --checkpoint-verification, --pricing and --commit')
            prepared = planning.prepare(args.prepare, source_plan_path=args.source_plan,
                checkpoint_verification_path=args.checkpoint_verification, pricing_file=args.pricing, commit=args.commit)
            result = {'status': prepared['status'], 'api_allowed': False,
                'plan_path': prepared['identity']['plan_path'], 'plan_identity_hash': prepared['identity_hash'],
                'unattempted_requests': 17, 'http520_retries': 0, 'research_api_requests_sent': 0}
        else:
            if not args.plan:
                parser.error('--plan is required')
            if args.refresh_pricing:
                result = planning.refresh_pricing(args.plan, pricing_file=args.refresh_pricing)
            elif args.checkpoint:
                plan = planning.load_plan(args.plan); identity = plan['identity']
                directory = runner.base._checked_path(identity['enabled_directory'])
                if not directory.is_dir() or not all((directory / p).is_file() for p in ('approval.json','enabled.json','ledger.json')):
                    raise GenerationError('goal006c_not_enabled_checkpoint_cannot_create_state')
                with runner.source_locked(identity), runner.base.run_lock(directory), runner.base._authority_locked(identity):
                    directory, ledger = runner.load_enabled(plan)
                    result = runner._save(plan, directory, ledger, 'manual_save_only', None)
            elif args.execute:
                plan = planning.load_plan(args.plan)
                if not all(bool(os.environ.get(key)) for key in providers.KEY_NAMES.values()):
                    raise GenerationError('both_provider_keys_required_no_key_probe_performed')
                planning.current_pricing(plan['identity'], fresh=True)
                if not runner.status(args.plan)['approved']:
                    runner.initialize_enabled(args.plan, approval=tty_approval(plan))
                result = runner.run(args.plan, permission=True)
            else:
                result = runner.status(args.plan); plan = planning.load_plan(args.plan)
                try:
                    planning.current_pricing(plan['identity'], fresh=True)
                    result['pricing_current_utc_day'] = True
                except GenerationError:
                    result['pricing_current_utc_day'] = False
                result['required_new_approval_scope'] = runner.APPROVAL_SCOPE
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
        return 6 if result.get('halt_reason') or result.get('operational_stop') or result.get('status') == 'SAVE_FAILED' else 0
    except (archive.RecoveryError, OriginalRecoveryError) as exc:
        print(json.dumps({'status': 'STOPPED', 'error': exc.report}), file=sys.stderr)
        return 5
    except GenerationError as exc:
        print(json.dumps({'status': 'STOPPED', 'code': str(exc)}), file=sys.stderr)
        return 5
    except Exception as exc:
        print(json.dumps({'status': 'STOPPED', 'code': type(exc).__name__}), file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
