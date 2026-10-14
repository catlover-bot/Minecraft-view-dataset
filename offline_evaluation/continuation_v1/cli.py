"""One continuation scope, exact real-TTY approval and same-ledger operations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from ..resmoke_v2.cli import _confirm_from_tty, TTYConfirmationError
from ..server_v2.generation import GenerationError, utc_now
from ..server_v2.providers import KEY_NAMES
from . import plan as planning


def confirmation_text(plan):
    return f"APPROVE GOAL006 {plan['identity_hash'][:12]} 800 SENDS 0 RETRIES USD30 TOTAL"


def preview(plan):
    from . import runner
    identity = plan['identity']
    current = runner.status(identity['plan_path'])
    pricing = planning.current_pricing(identity)
    return {'status': current['status'], 'api_allowed': False,
        'plan_identity_hash': plan['identity_hash'], 'same_fixed_experiment_continuation': True,
        'semantic_protocol_id': identity['semantic_protocol_id'], 'run_protocol_id': identity['protocol_id'],
        'remaining_buildings': 50, 'planned_inputs': 800, 'global_schedule_range_inclusive': [160, 959],
        'new_max_calls': 800, 'cumulative_max_calls': 990,
        'new_retries': 0, 'cumulative_retries': 4, 'output_cap': 32768,
        'prior_usd_upper_bound': 4.048922, 'cumulative_usd_proposal': 30,
        'new_scope_approved': current['approved'], 'budgets': identity['budgets'],
        'batch_scene_ids': identity['batch_scene_ids'], 'segments': identity['segments'],
        'usage': current['used'], 'attempted_inputs': current['attempted_inputs'],
        'not_run_inputs': current['not_run_inputs'], 'halt_reason': current['halt_reason'],
        'pricing_verified_on': {p: r['verified_on'] for p, r in pricing.items()},
        'paid_start_requires_current_utc_date': utc_now()[:10],
        'api_key_present': {p: bool(os.environ.get(key)) for p, key in KEY_NAMES.items()},
        'canonical_output_directory': identity['enabled_directory'],
        'resume_command': runner.resume_command(plan),
        'cost_semantics': 'USD30 is cumulative including old consumption; not an additional USD30, invoice, or completion guarantee'}


def launch(plan_path, *, preview_only=False, github_checkpoints=False):
    from . import runner
    plan = planning.load_plan(plan_path)
    display = preview(plan)
    print(json.dumps(display, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    if preview_only:
        print('PREVIEW_ONLY: additional research sends=0; approval not requested')
        return 0
    identity = plan['identity']
    planning.verify_parent(identity['parent'])
    if display['status'] == 'COMPLETE_CONTINUATION800':
        print('ALREADY_COMPLETE: additional research sends=0')
        return 0
    if not all(display['api_key_present'].values()):
        print('MISSING_API_KEY: set both keys in this executing terminal; values are never printed')
        return 3
    if display['new_scope_approved']:
        print('RESUME: original approval and ledger; only unattempted suffix inputs; uncertain attempts never resend')
    else:
        planning.current_pricing(identity, fresh=True)
        if not _confirm_from_tty(confirmation_text(plan)):
            print('NOT_APPROVED: additional research sends=0; no enabled run created')
            return 2
        runner.initialize_enabled(plan_path, approval={
            'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
            'enabled_directory': identity['enabled_directory'], 'approved_at_utc': utc_now(),
            'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE})
    result = runner.run(plan_path, permission=True, github_checkpoints=github_checkpoints)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result.get('operational_stop'):
        return 6
    return 0 if result['experiment_complete'] else 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prepare = subs.add_parser('prepare', help='CPU only: unique disabled suffix800 plan')
    for name in ('destination', 'pricing', 'commit', 'checkpoint-sources'):
        prepare.add_argument('--' + name, required=True)
    launch_parser = subs.add_parser('launch', help='Preview then real-TTY approval; same-ledger resume')
    launch_parser.add_argument('--plan', required=True)
    launch_parser.add_argument('--preview', action='store_true')
    launch_parser.add_argument('--github-checkpoints', action='store_true')
    for name in ('preview', 'status', 'evaluate'):
        command = subs.add_parser(name)
        command.add_argument('--plan', required=True)
    refresh = subs.add_parser('refresh-pricing')
    refresh.add_argument('--plan', required=True)
    refresh.add_argument('--pricing', required=True)
    aggregate = subs.add_parser('aggregate', help='CPU only: parent160 plus suffix800; inference only when fixed960 complete')
    aggregate.add_argument('--plan', required=True)
    aggregate.add_argument('--destination', required=True)
    checkpoint = subs.add_parser('checkpoint', help='Snapshot existing state without invoking generation')
    checkpoint.add_argument('--plan', required=True)
    checkpoint.add_argument('--destination', required=True)
    checkpoint.add_argument('--github', action='store_true')
    upload = subs.add_parser('checkpoint-upload', help='Retry only saved checkpoint upload; never generate')
    upload.add_argument('--checkpoint', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = planning.prepare(args.destination, pricing_file=args.pricing,
                                      commit=args.commit, checkpoint_sources=args.checkpoint_sources)
            result = preview(result)
        elif args.command == 'preview':
            result = preview(planning.load_plan(args.plan))
        elif args.command == 'status':
            from .runner import status
            result = status(args.plan)
        elif args.command == 'evaluate':
            from .runner import evaluate_and_summarize
            result = evaluate_and_summarize(args.plan)
        elif args.command == 'refresh-pricing':
            result = planning.refresh_pricing(args.plan, pricing_file=args.pricing)
        elif args.command == 'aggregate':
            from .aggregate import collect
            result = collect(args.plan, args.destination)
        elif args.command == 'checkpoint':
            from .checkpoint import create
            result = create(args.plan, destination=args.destination, github=args.github)
        elif args.command == 'checkpoint-upload':
            from .checkpoint import upload
            result = upload(args.checkpoint)
        else:
            return launch(args.plan, preview_only=args.preview, github_checkpoints=args.github_checkpoints)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (GenerationError, ValueError, OSError) as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
