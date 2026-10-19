"""Prepare a disabled eight-row child without changing the parent or authority."""
from decimal import Decimal
import os
from pathlib import Path

from . import runtime as r


def prepare_plan(parent_plan_path, child_root, *, parent_evaluation, github_verification,
                 parent_checkpoint, parent_archive, specification, implementation_hashes,
                 pricing_evidence, parent_evaluation_summary=None, fixture_only=False):
    root = r.path_safe(child_root)
    r.require(root.is_dir() and not any(os.path.lexists(root / name) for name in ('plan.json', 'enabled', 'execution.lock')),
              'new child plan root required; never reset existing execution')
    parent = r.frozen.verify_plan(parent_plan_path)
    parent_root = Path(parent['canonical_root'])
    if not fixture_only:
        r.require(root.parent == Path('/root/minecraft-runs') and root.name.startswith('goal015a-'), 'canonical Goal015A root required')
        r.require(not os.path.lexists(r.EXECUTION_BINDING), 'prepared continuation already registered')
        for candidate in root.parent.glob('goal015a-*/plan.json'):
            r.require(r.read_json(candidate).get('parent_plan_id') != parent['plan_id'], 'continuation of this parent already prepared')
    refs = {'plan': r.descriptor(parent_plan_path), 'approval': r.descriptor(parent_root / 'enabled/approval.json'),
            'authority': r.descriptor(parent_root / 'enabled/authority.json'), 'ledger': r.descriptor(parent_root / 'enabled/ledger.json'),
            'execution_lock': r.descriptor(parent_root / 'execution.lock'), 'evaluation': parent_evaluation,
            'github_verification': github_verification, 'checkpoint': parent_checkpoint, 'archive': parent_archive}
    if parent_evaluation_summary is not None:
        refs['evaluation_summary'] = parent_evaluation_summary
    ledger = r.bound(refs['ledger'])
    carry = r.frozen.accounting(ledger)['known_cost_usd']
    remaining = [original for original, observed in zip(parent['schedule'], ledger['calls']) if observed['state'] == 'NOT_SENT']
    total_reserve = sum((r._decimal(row['reserve_usd']) for row in remaining), Decimal(0))
    r.require(total_reserve == Decimal('2.117504'), 'fixed remaining conservative reservation differs')
    plan = {'schema': 'goal015a_disabled_plan_v1', 'canonical_root': str(root), 'api_allowed': False,
            'budget_approved': False, 'parent_plan_id': parent['plan_id'], 'parent': refs,
            'parent_known_cost_usd': format(r._decimal(carry), '.8f'), 'proposed_budget_usd': '7.00',
            'planned_calls': 8, 'total_planned_calls': 80, 'planned_buildings': 1, 'total_planned_buildings': 10,
            'concurrency': 1, 'max_retries': 0, 'timeout_seconds': parent['timeout_seconds'],
            'schedule': remaining, 'pricing_contract': parent['pricing_contract'], 'pricing_evidence': pricing_evidence,
            'implementation_hashes': implementation_hashes, 'specification': specification,
            'costs': {'remaining_reservation_usd': str(total_reserve),
                      'parent_plus_full_remaining_reservation_usd': str(r._decimal(carry) + total_reserve),
                      'child_maximum_remaining_cap_usd': str(Decimal('7.00') - r._decimal(carry)),
                      'additional_approved_cap_proposal_usd': '2.00'}}
    if fixture_only:
        plan['fixture_only'] = True
    with r.paired_locks(plan):
        r.verify_parent(plan)
        r.pricing_fresh(plan, pricing_evidence)
        for name, expected in implementation_hashes.items():
            r.require(r.descriptor(name)['sha256'] == expected, 'implementation hash differs before preparation')
        r.require(r.descriptor(specification['path']) == specification, 'specification differs before preparation')
        plan['plan_id'] = r.identity(plan)
        r._save(root / 'plan.json', plan, exclusive=True)
    return {'plan': r.descriptor(root / 'plan.json'), 'plan_id': plan['plan_id'],
            'parent_plan_id': plan['parent_plan_id'], 'canonical_root': str(root)}
