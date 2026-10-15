"""Prepare or preview by default; supplemental POSTs require a new exact TTY approval."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from offline_evaluation.operations.goal006b import planning, runner
from offline_evaluation.operations.goal006a.archive import RecoveryError
from offline_evaluation.server_v2.generation import GenerationError, utc_now
from offline_evaluation.server_v2 import providers


def confirmation_text(plan):
    return ('APPROVE GOAL006B ' + plan['identity_hash']
            + ' 19 NEW + 1 AUTH RETRY; MAX 991 CALLS; USD30; KEEP UNKNOWN RESERVATION')


def tty_approval(plan):
    from contextlib import ExitStack

    message = ('Goal006Bは別台帳で最大20回を送信します。未実行19件とHTTP401の明示再試行1件です。\n'
       '元の401・halt・usage不明予約を維持します。予約額は請求額ではありません。\n'
       '累積送信上限991、累積USD30、自動retry0。上限の引上げはこの認証再試行1回だけです。\n'
       'この実行は料金が発生し得ます。承認する場合だけ次の全文を入力してください:\n'
       + confirmation_text(plan) + '\n> ')
    try:
        with ExitStack() as handles:
            try:
                reader = handles.enter_context(open('/dev/tty', 'r', encoding='utf-8', newline=''))
                writer = handles.enter_context(open('/dev/tty', 'w', encoding='utf-8', buffering=1))
            except OSError:
                raise GenerationError('supplemental_tty_unavailable_no_activation') from None
            if not reader.isatty() or not writer.isatty():
                raise GenerationError('supplemental_real_tty_required_no_activation')
            writer.write(message)
            writer.flush()
            answer = reader.readline()
    except (OSError, UnicodeError):
        raise GenerationError('supplemental_tty_io_error_no_activation') from None
    except EOFError:
        raise GenerationError('supplemental_approval_eof_no_activation') from None
    except KeyboardInterrupt:
        raise GenerationError('supplemental_approval_interrupted_no_activation') from None
    if answer == '':
        raise GenerationError('supplemental_approval_eof_no_activation')
    if answer != confirmation_text(plan) + '\n':
        raise GenerationError('supplemental_exact_approval_not_given_no_activation')
    return {'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
            'enabled_directory': plan['identity']['enabled_directory'], 'approved_at_utc': utc_now(),
            'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE,
            'acknowledges_unknown_hold': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', metavar='NEW_DIRECTORY')
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--refresh-pricing', metavar='OFFICIAL_PRICING_JSON')
    mode.add_argument('--checkpoint', action='store_true', help='Save a local snapshot only; no generation or GitHub upload')
    parser.add_argument('--source-plan', default='/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json')
    parser.add_argument('--checkpoint-verification')
    parser.add_argument('--commit')
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            if args.plan or not args.checkpoint_verification or not args.commit:
                parser.error('--prepare requires --checkpoint-verification and --commit, without --plan')
            prepared = planning.prepare(args.prepare, source_plan_path=args.source_plan,
                   checkpoint_verification_path=args.checkpoint_verification, commit=args.commit)
            result = {'status': prepared['status'], 'api_allowed': False,
                      'plan_path': prepared['identity']['plan_path'], 'plan_identity_hash': prepared['identity_hash'],
                      'manual_auth_retry_requests': 1, 'unattempted_requests': 19,
                      'research_api_requests_sent': 0}
        else:
            if not args.plan:
                parser.error('--plan is required')
            if args.refresh_pricing:
                result = planning.refresh_pricing(args.plan, pricing_file=args.refresh_pricing)
            elif args.checkpoint:
                plan = planning.load_plan(args.plan)
                directory = runner.base._checked_path(plan['identity']['enabled_directory'])
                if not directory.is_dir() or not all((directory / name).is_file()
                        for name in ('approval.json', 'enabled.json', 'ledger.json')):
                    raise GenerationError('supplement_not_enabled_checkpoint_cannot_create_state')
                with runner.source_locked(plan['identity']), runner.base.run_lock(directory), runner.base._authority_locked(plan['identity']):
                    directory, ledger = runner.load_enabled(plan)
                    result = runner._save(plan, directory, ledger, 'manual_local_checkpoint', None)
            elif args.execute:
                plan = planning.load_plan(args.plan)
                if not all(bool(os.environ.get(key)) for key in providers.KEY_NAMES.values()):
                    raise GenerationError('both_provider_keys_required_no_key_probe_performed')
                planning.current_pricing(plan['identity'], fresh=True)
                state = runner.status(args.plan)
                if not state['approved']:
                    runner.initialize_enabled(args.plan, approval=tty_approval(plan))
                result = runner.run(args.plan, permission=True)
            else:
                result = runner.status(args.plan)
                plan = planning.load_plan(args.plan)
                try:
                    planning.current_pricing(plan['identity'], fresh=True)
                    result['pricing_current_utc_day'] = True
                except GenerationError:
                    result['pricing_current_utc_day'] = False
                result['required_new_approval_scope'] = runner.APPROVAL_SCOPE
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
        return 6 if result.get('halt_reason') or result.get('operational_stop') or result.get('status') == 'SAVE_FAILED' else 0
    except RecoveryError as exc:
        print(json.dumps({'status': 'STOPPED', 'error': exc.report}), file=sys.stderr)
        return 5
    except GenerationError as exc:
        # Local validators use fixed diagnostic codes, never provider bodies.
        print(json.dumps({'status': 'STOPPED', 'code': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 5
    except Exception as exc:
        print(json.dumps({'status': 'STOPPED', 'code': type(exc).__name__}), file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
