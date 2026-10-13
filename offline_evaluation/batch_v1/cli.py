"""First evaluation batch preview, exact terminal approval and same-ledger resume."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

from ..server_v2.generation import GenerationError, utc_now
from ..server_v2.providers import KEY_NAMES
from ..resmoke_v2.cli import _confirm_from_tty, TTYConfirmationError
from . import plan as planning


def confirmation_text(plan):
    return f"APPROVE GOAL005 {plan['identity_hash'][:12]} 160 SENDS 0 RETRIES USD10 TOTAL"


def preview(plan):
    from . import runner
    identity = plan['identity']
    used = {'prior': identity['parent']['usage'],
            'new': {'calls': 0, 'retries': 0, 'tokens': 0, 'cost_usd_upper_bound': 0,
                    'unknown_usage_attempts': 0},
            'combined': dict(identity['parent']['usage'])}
    enabled = Path(identity['enabled_directory'])
    if enabled.exists():
        _, ledger = runner._load_enabled(plan)
        used = runner.budget_totals(identity, ledger)
    pricing = planning.current_pricing(identity)
    return {'status': plan['status'], 'api_allowed': False, 'plan_identity_hash': plan['identity_hash'],
        'protocol_id': identity['protocol_id'], 'scope': identity['scope'],
        'fixed_targets_in_send_order': [{k: row[k] for k in (
            'schedule_index', 'scene_id', 'condition', 'description_model', 'provider', 'model', 'input_id', 'input_sha256')}
            for row in identity['schedule']],
        'batch_scene_ids': identity['batch_scene_ids'], 'planned_inputs': len(identity['schedule']),
        'fixed_full_evaluation_buildings': len(identity['evaluation_scene_ids']),
        'remaining_800_authorized': False, 'schema_sha256': identity['schema_sha256'],
        'prompt_sha256': identity['prompt_sha256'], 'budgets': identity['budgets'],
        'output_cap_cpu_assessment': identity.get('output_cap_cpu_assessment'),
        'usage': used, 'remaining_theoretical_usd': float(Decimal(str(identity['budgets']['max_cost_usd']))
                                                        - Decimal(str(used['combined']['cost_usd_upper_bound']))),
        'reservation': identity['reservation_policy'],
        'pricing_verified_on': {p: r['verified_on'] for p, r in pricing.items()},
        'paid_start_requires_current_utc_date': utc_now()[:10],
        'api_key_present': {p: bool(os.environ.get(key)) for p, key in KEY_NAMES.items()},
        'canonical_output_directory': identity['enabled_directory'],
        'authority_path': identity['authority_path'], 'approval_already_exists': enabled.exists(),
        'cost_semantics': 'application conservative stopping limits; not invoice or completion cost prediction'}


def launch(plan_path, *, preview_only=False):
    from . import runner
    plan = planning.load_plan(plan_path)
    display = preview(plan)
    print(json.dumps(display, ensure_ascii=False, sort_keys=True, indent=2))
    if preview_only:
        print('PREVIEW_ONLY: additional research sends=0; approval not requested')
        return 0
    identity = plan['identity']
    planning.verify_parent(identity['parent'])
    planning.current_pricing(identity, fresh=True)
    if not all(display['api_key_present'].values()):
        print('MISSING_API_KEY: set both keys in this terminal; values are never printed')
        return 3
    if Path(identity['enabled_directory']).exists():
        print('RESUME: original approval and ledger; only unattempted inputs; uncertain attempts never resend')
    else:
        if not _confirm_from_tty(confirmation_text(plan)):
            print('NOT_APPROVED: additional research sends=0; no enabled run created')
            return 2
        approval = {'manual_confirmed': True, 'plan_identity_hash': plan['identity_hash'],
            'enabled_directory': identity['enabled_directory'], 'approved_at_utc': utc_now(),
            'confirmation_channel': '/dev/tty', 'scope': runner.APPROVAL_SCOPE}
        runner.initialize_enabled(plan_path, approval=approval)
    result = runner.run(plan_path, permission=True)
    summary = runner.evaluate_and_summarize(plan_path)
    print(json.dumps({'generation': result, 'evaluation_summary': summary}, ensure_ascii=False, indent=2))
    return 0 if summary['experiment_complete'] else 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prepare = subs.add_parser('prepare', help='CPU only: create a unique disabled batch plan')
    prepare.add_argument('--destination', required=True)
    prepare.add_argument('--pricing', required=True)
    prepare.add_argument('--commit', required=True)
    start = subs.add_parser('launch', help='Preview then exact /dev/tty approval; resume same ledger')
    start.add_argument('--plan', required=True)
    start.add_argument('--preview', action='store_true')
    for name in ('show', 'evaluate'):
        command = subs.add_parser(name)
        command.add_argument('--plan', required=True)
    refresh = subs.add_parser('refresh-pricing', help='Preserve ledger/approval; append actual official re-verification')
    refresh.add_argument('--plan', required=True)
    refresh.add_argument('--pricing', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = planning.prepare(args.destination, pricing_file=args.pricing, commit=args.commit)
            print(json.dumps(preview(result), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'show':
            print(json.dumps(preview(planning.load_plan(args.plan)), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'evaluate':
            from .runner import evaluate_and_summarize
            print(json.dumps(evaluate_and_summarize(args.plan), ensure_ascii=False, indent=2))
            return 0
        if args.command == 'refresh-pricing':
            print(json.dumps(planning.refresh_pricing(args.plan, pricing_file=args.pricing), indent=2))
            return 0
        return launch(args.plan, preview_only=args.preview)
    except (GenerationError, ValueError) as exc:
        print(f'STOPPED: {exc}', file=sys.stderr)
        return 5


if __name__ == '__main__':
    raise SystemExit(main())
