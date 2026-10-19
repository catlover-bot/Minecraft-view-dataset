"""Eight-call continuation with immutable parent accounting and two locks.

Uses the frozen Goal015 transport and durable-file helpers. The parent plan,
authority, ledger and responses are never written by this module.
"""
from contextlib import contextmanager, ExitStack
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import subprocess

from ...experiments.goal015 import runtime as frozen

RuntimeErrorSafe = frozen.RuntimeErrorSafe
require, descriptor, read_json = frozen.require, frozen.descriptor, frozen.read_json
path_safe, identity, bound = frozen.path_safe, frozen.identity, frozen.bound
_save, _decimal, now = frozen._save, frozen._decimal, frozen.now
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_BINDING = REPOSITORY_ROOT / 'reports/server_only/goal015a/EXECUTION_BINDING.json'
PARENT_ID = '5b80fdef342dd2b9f68b3c2bec2d3d83af2731c3b5a9c8940959846ea11c68ce'
PARENT_SHA = '6533090e07090b7279f0383937631c609b17c84c922f8cd8deadcd2b9ea22257'
CALL_IDS = [f'goal015_call_{i:03d}' for i in range(73, 81)]
MODEL_FAILURES = frozenset(('parse_failure', 'validation_failure', 'execution_failure', 'truncation', 'refusal'))
CONTRACT_FAILURES = frozenset(('response_model_mismatch', 'unexpected_response_feature', 'incomplete_response'))


def verify_parent(plan):
    """Authenticate the fixed stopped parent, including all selected raw files."""
    refs = plan['parent']
    for ref in refs.values():
        require(descriptor(ref['path']) == ref, 'bound parent artifact changed')
    parent = frozen.verify_plan(refs['plan']['path'])
    fixture = plan.get('fixture_only') is True
    require(parent.get('fixture_only') is True if fixture else parent.get('fixture_only') is not True,
            'parent/child fixture scope differs')
    if not fixture:
        require(parent['plan_id'] == PARENT_ID and refs['plan']['sha256'] == PARENT_SHA,
                'fixed original parent identity required')
    require(parent['plan_id'] == plan['parent_plan_id'], 'parent plan identity changed')
    root = Path(parent['canonical_root'])
    for field, relative in (('plan', 'plan.json'), ('approval', 'enabled/approval.json'),
                            ('authority', 'enabled/authority.json'), ('ledger', 'enabled/ledger.json'),
                            ('execution_lock', 'execution.lock')):
        require(Path(refs[field]['path']) == root / relative, 'canonical parent artifact required')
    approval, authority, ledger = (bound(refs[k]) for k in ('approval', 'authority', 'ledger'))
    require(authority.get('status') == 'APPROVED' and authority.get('scope_calls') == 80
            and authority.get('max_retries') == 0 and authority['plan_id'] == parent['plan_id'] and authority['canonical_root'] == str(root)
            and authority['plan'] == refs['plan'] and authority['approval'] == refs['approval'],
            'parent authority identity changed')
    frozen._validate_ledger(parent, ledger, approval, for_resume=True)
    require(ledger['status'] == 'SAFE_STOP' and ledger.get('halt', {}).get('reason') ==
            'next complete paired block exceeds approved cap', 'fixed parent budget stop required')
    require([row['state'] for row in ledger['calls']] == ['SETTLED'] * 72 + ['NOT_SENT'] * 8,
            'parent must remain72 settled and8 unsent with no unknowns')
    costs = frozen.accounting(ledger)
    require(_decimal(costs['unsettled_and_unknown_holds_usd']) == 0
            and _decimal(costs['known_cost_usd']) == _decimal(plan['parent_known_cost_usd']),
            'parent carried accounting changed')
    require(_decimal(plan['parent_known_cost_usd']) == Decimal('4.43765700'), 'fixed parent charge required')
    require(plan['schedule'] == parent['schedule'][72:] and
            [row['call_id'] for row in plan['schedule']] == CALL_IDS, 'only original unsent073 through080 allowed')
    for row in plan['schedule']:
        require(descriptor(row['payload']['path']) == row['payload'], 'original payload bytes changed')
    evaluation = bound(refs['evaluation'])
    require(evaluation.get('plan_binding', {}).get('plan') == refs['plan']
            and evaluation['plan_binding'].get('ledger') == refs['ledger'], 'parent evaluation source binding differs')
    require([row['call_id'] for row in evaluation['rows']] == [row['call_id'] for row in parent['schedule']],
            'parent evaluation must retain original80 rows')
    proof = bound(refs['github_verification'])
    require(proof.get('status') == 'GITHUB_VERIFIED' and proof.get('draft') is True,
            'verified parent Draft save required before continuation')
    require(type(proof.get('release_id')) is int and type(proof.get('asset_id')) is int,
            'parent numeric release and asset IDs required')
    if not fixture:
        require(proof.get('fixture_only') is not True and proof.get('github_transport') == 'GH_LIVE_SAVE_ONLY',
                'real parent GitHub verification required')
    checkpoint = bound(refs['checkpoint'])
    require(checkpoint['plan_id'] == parent['plan_id'] and checkpoint['archive_sha256'] ==
            refs['archive']['sha256'] == proof['archive_sha256'], 'parent archive/save proof differs')
    require(plan['pricing_contract'] == parent['pricing_contract'], 'parent pricing contract changed')
    return {'plan': parent, 'ledger': ledger, 'approval': approval, 'authority': authority}


def verify_plan(plan_path):
    path = path_safe(plan_path)
    plan = read_json(path)
    root = path_safe(plan['canonical_root'])
    require(path == root / 'plan.json', 'copied child plan cannot activate')
    require(plan.get('schema') == 'goal015a_disabled_plan_v1' and plan.get('api_allowed') is False
            and plan.get('budget_approved') is False and plan.get('plan_id') == identity(plan),
            'immutable disabled child plan required')
    require(plan['proposed_budget_usd'] == '7.00' and plan['planned_calls'] == 8
            and plan['total_planned_calls'] == 80 and plan['concurrency'] == 1
            and plan['planned_buildings'] == 1 and plan['total_planned_buildings'] == 10
            and plan['max_retries'] == 0 and plan['timeout_seconds'] == 600, 'fixed child scope and combined cap required')
    if plan.get('fixture_only') is not True:
        require(root.parent == Path('/root/minecraft-runs') and root.name.startswith('goal015a-'),
                'dedicated canonical child root required')
        registry = read_json(EXECUTION_BINDING)
        require(registry['plan_id'] == plan['plan_id'] and registry['canonical_root'] == str(root)
                and registry['plan'] == descriptor(path) and registry['parent_plan_id'] == plan['parent_plan_id'],
                'prepared child registry differs; copy/replay refused')
    for path_string, expected in plan['implementation_hashes'].items():
        require(descriptor(path_string)['sha256'] == expected, 'child implementation changed')
    require(descriptor(plan['specification']['path']) == plan['specification'], 'child specification changed')
    require(descriptor(plan['pricing_evidence']['path']) == plan['pricing_evidence'], 'bound pricing evidence changed')
    verify_parent(plan)
    for offset in range(0, 8, 2):
        a, b = plan['schedule'][offset:offset + 2]
        require(a['paired_block_id'] == b['paired_block_id'] and {a['condition'], b['condition']} ==
                {'SUMMARY_ONLY', 'FULL_RECORD'}, 'original adjacent S/F pair required')
    return plan


def pricing_fresh(plan, evidence):
    require(evidence is not None, 'explicit current parent-contract pricing evidence required')
    evidence = evidence if isinstance(evidence, dict) else descriptor(evidence)
    parent = bound(plan['parent']['plan'])
    frozen.pricing_fresh(parent, evidence)
    return evidence


def approval_text(plan):
    return (f"APPROVE GOAL015A {plan['plan_id']} PARENT {plan['parent_plan_id']} "
            'COMBINED USD 7.00 ADDITIONAL CAP USD 2.00 REMAINING 8 CALLS TOTAL 80 RETRY 0')


def confirm_tty(plan):
    expected = approval_text(plan)
    try:
        with open('/dev/tty', 'r', encoding='utf-8') as reader, open('/dev/tty', 'w', encoding='utf-8') as writer:
            require(os.isatty(reader.fileno()) and os.isatty(writer.fileno()), 'real controlling TTY required')
            writer.write(f"Parent known charge USD {plan['parent_known_cost_usd']}; combined cap USD 7.00; "
                         f"cap increase USD 2.00; remaining8; total80;retry0\nType exactly: {expected}\n> ")
            writer.flush()
            answer = reader.readline(len(expected) + 4)
            require(answer != '' and answer.rstrip('\r\n') == expected, 'explicit exact child approval absent')
    except RuntimeErrorSafe:
        raise
    except (OSError, UnicodeError, ValueError):
        raise RuntimeErrorSafe('TTY unavailable or read/write failed') from None
    return True


@contextmanager
def paired_locks(plan, *, create_child=False):
    """Parent existing lock is opened rb: its bytes and namespace stay unchanged."""
    with ExitStack() as stack:
        parent = stack.enter_context(path_safe(plan['parent']['execution_lock']['path']).open('rb'))
        try:
            fcntl.flock(parent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeErrorSafe('parent execution already active') from None
        stack.callback(fcntl.flock, parent.fileno(), fcntl.LOCK_UN)
        child_path = path_safe(Path(plan['canonical_root']) / 'execution.lock')
        if create_child:
            stack.enter_context(frozen.canonical_lock(plan['canonical_root']))
        elif child_path.exists():
            child = stack.enter_context(child_path.open('rb'))
            try:
                fcntl.flock(child.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeErrorSafe('child execution already active') from None
            stack.callback(fcntl.flock, child.fileno(), fcntl.LOCK_UN)
        yield


def accounting(plan, ledger):
    child = frozen.accounting(ledger)
    carried = _decimal(plan['parent_known_cost_usd'])
    return dict(child, child_known_cost_usd=child['known_cost_usd'], parent_known_cost_usd=str(carried),
                combined_known_cost_usd=str(carried + _decimal(child['known_cost_usd'])),
                combined_committed_usd=str(carried + _decimal(child['committed_usd'])), combined_cap_usd='7.00')


def _validate_ledger(plan, ledger, approval, *, for_resume=False):
    require(ledger['plan_id'] == approval['plan_id'] == plan['plan_id'] and
            approval['parent_plan_id'] == plan['parent_plan_id'] and approval['canonical_root'] == plan['canonical_root']
            and approval['approved_combined_budget_usd'] == '7.00' and approval['scope_calls'] == 8
            and approval['parent_known_cost_usd'] == plan['parent_known_cost_usd'] and approval['status'] == 'APPROVED',
            'same child combined-budget authority required')
    require(approval.get('approval_text') == approval_text(plan), 'child approval text differs')
    require(len(ledger['calls']) == 8, 'eight child ledger rows required')
    if for_resume:
        require(not ledger.get('requires_new_explicit_approval') and not any(
            row.get('failure_kind') in CONTRACT_FAILURES for row in ledger['calls']), 'contract failure requires new explicit approval')
    for original, row in zip(plan['schedule'], ledger['calls']):
        require(all(row.get(k) == v for k, v in original.items()), 'child ledger original schedule changed')
        require(row['state'] in ('NOT_SENT', 'SENT_PENDING', 'UNKNOWN_USAGE', 'SETTLED'), 'invalid child state')
        reserve = _decimal(row['reservation_usd'])
        require(reserve in (Decimal(0), _decimal(row['reserve_usd'])), 'child reservation changed')
        if row['state'] == 'NOT_SENT':
            require(row['raw'] is None and row['send_status'] == 'NOT_SENT' and row.get('response_text_path') is None
                    and row.get('response_received') is False and row.get('completed_response') is False
                    and 'started_at_utc' not in row and 'finished_at_utc' not in row,
                    'NOT_SENT has transmission evidence')
            responses = Path(plan['canonical_root']) / 'enabled/responses'
            require(not any(os.path.lexists(responses / (row['call_id'] + suffix)) for suffix in ('.raw.json', '.text.txt')),
                    'NOT_SENT has existing response files; restored state cannot resend')
        else:
            require(reserve == _decimal(row['reserve_usd']), 'whole attempted reservation required')
        if row['state'] == 'SETTLED':
            require(row['usage_known'] is True and row['known_cost_usd'] is not None and row['raw'] is not None,
                    'settlement requires durable raw and known cost')
            cost = _decimal(row['known_cost_usd'])
            if for_resume:
                require(cost <= reserve, 'observed cost exceeds reservation')
        else:
            require(row['usage_known'] is False and row['known_cost_usd'] is None, 'unsettled cost cannot be released')
        for key in ('raw', 'response_text_path'):
            if row.get(key) is not None:
                ref = row[key]
                require(path_safe(ref['path']).is_relative_to(Path(plan['canonical_root']) / 'enabled/responses')
                        and descriptor(ref['path']) == ref, 'child response evidence changed or escaped canonical root')
    if for_resume:
        require(_decimal(accounting(plan, ledger)['combined_committed_usd']) <= Decimal('7.00'), 'combined cap exceeded')


def preview(plan_path):
    plan = verify_plan(plan_path)
    return {'status': 'REMAINING8_READY_AWAITING_EXPLICIT_APPROVAL', 'plan_id': plan['plan_id'],
            'parent_plan_id': plan['parent_plan_id'], 'planned_calls': 8, 'total_planned_calls': 80,
            'parent_known_cost_usd': plan['parent_known_cost_usd'], 'proposed_combined_cap_usd': '7.00',
            'budget_approved': False, 'api_allowed': False, 'model_account_availability': 'NOT_RUN',
            'concurrency': 1, 'max_retries': 0, 'research_api_requests_sent': 0, 'approval_text': approval_text(plan)}


def status(plan_path):
    plan = verify_plan(plan_path)
    enabled = Path(plan['canonical_root']) / 'enabled'
    if not enabled.exists():
        return preview(plan_path)
    return _result(plan, verify_enabled(plan)['ledger'])


def verify_enabled(plan, *, for_resume=False):
    """Read-only authentication shared by continuation, evaluation and saving."""
    enabled = Path(plan['canonical_root']) / 'enabled'
    approval, authority, ledger = (read_json(enabled / name) for name in ('approval.json', 'authority.json', 'ledger.json'))
    require(authority.get('status') == 'APPROVED' and authority.get('plan_id') == plan['plan_id'] and authority.get('parent_plan_id') == plan['parent_plan_id']
            and authority.get('canonical_root') == plan['canonical_root'] and authority.get('scope_calls') == 8
            and authority.get('max_retries') == 0 and authority.get('plan') == descriptor(enabled.parent / 'plan.json')
            and authority.get('approval') == descriptor(enabled / 'approval.json')
            and authority.get('parent_ledger') == plan['parent']['ledger'], 'child authority binding changed')
    _validate_ledger(plan, ledger, approval, for_resume=for_resume)
    return {'approval': approval, 'authority': authority, 'ledger': ledger}


def _result(plan, ledger):
    return {'status': ledger['status'], 'plan_id': plan['plan_id'], 'budget': accounting(plan, ledger),
            'calls_not_sent': sum(row['state'] == 'NOT_SENT' for row in ledger['calls']),
            'calls_uncertain': sum(row['state'] in ('SENT_PENDING', 'UNKNOWN_USAGE') for row in ledger['calls']),
            'calls_settled': sum(row['state'] == 'SETTLED' for row in ledger['calls']),
            'checkpoint': ledger.get('last_checkpoint'), 'failure': ledger.get('halt'),
            'fixture_only': plan.get('fixture_only') is True}


def execute(plan_path, *, pricing_evidence, resume=False, adapter=None, confirmation=None, checkpoint_callback=None):
    plan = verify_plan(plan_path)
    root = Path(plan['canonical_root'])
    fixture = plan.get('fixture_only') is True
    require(fixture or (adapter is None and confirmation is None and checkpoint_callback is None), 'fixture hooks forbidden for production')
    require(not fixture or adapter is not None, 'fixture must supply a mock adapter')
    evidence = pricing_fresh(plan, pricing_evidence)
    if not fixture:
        committed = subprocess.check_output(['git', 'show', 'HEAD:reports/server_only/goal015a/EXECUTION_BINDING.json'],
                                            cwd=REPOSITORY_ROOT, stderr=subprocess.DEVNULL)
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPOSITORY_ROOT).decode().strip()
        require(committed == EXECUTION_BINDING.read_bytes() and branch == 'research/server-only-20261013',
                'committed child registry on dedicated branch required')
    enabled = root / 'enabled'
    if not resume:
        require(not os.path.lexists(enabled), 'existing child execution; explicit resume only')
        with paired_locks(plan):
            verify_parent(plan)
        require((confirmation or confirm_tty)(plan) is True, 'child approval refused')
    with paired_locks(plan, create_child=True):
        require(verify_plan(plan_path) == plan, 'child or parent changed while awaiting approval')
        pricing_fresh(plan, evidence)
        if resume:
            require(enabled.is_dir(), 'no approved child execution to resume')
            ledger = verify_enabled(plan, for_resume=True)['ledger']
        else:
            require(not os.path.lexists(enabled), 'duplicate child activation refused')
            enabled.mkdir(); frozen._fsync_directory(root)
            (enabled / 'responses').mkdir(); frozen._fsync_directory(enabled)
            approval = {'status': 'APPROVED', 'plan_id': plan['plan_id'], 'parent_plan_id': plan['parent_plan_id'],
                        'canonical_root': str(root), 'approved_combined_budget_usd': '7.00', 'scope_calls': 8,
                        'parent_known_cost_usd': plan['parent_known_cost_usd'], 'approved_at_utc': now(),
                        'approval_text': approval_text(plan), 'approval_method': 'FIXTURE_ONLY' if fixture else 'EXACT_REAL_TTY'}
            _save(enabled / 'approval.json', approval, exclusive=True)
            _save(enabled / 'authority.json', {'status': 'APPROVED', 'plan_id': plan['plan_id'], 'parent_plan_id': plan['parent_plan_id'],
                  'canonical_root': str(root), 'plan': descriptor(plan_path), 'approval': descriptor(enabled / 'approval.json'),
                  'parent_ledger': plan['parent']['ledger'], 'scope_calls': 8, 'max_retries': 0}, exclusive=True)
            ledger = {'schema': 'goal015a_execution_ledger_v1', 'plan_id': plan['plan_id'], 'status': 'APPROVED_NOT_SENT',
                      'calls': [dict(row, state='NOT_SENT', reservation_usd='0', known_cost_usd=None, usage_known=False,
                       send_status='NOT_SENT', response_received=False, completed_response=False, model_output_failure=False,
                       failure_kind=None, raw=None, response_text_path=None) for row in plan['schedule']]}
            _save(enabled / 'ledger.json', ledger, exclusive=True)
        ledger['pricing_evidence'] = evidence
        ledger['status'] = 'RUNNING'
        ledger.pop('halt', None)
        _save(enabled / 'ledger.json', ledger)
        if adapter is None:
            from ...experiments.goal015 import adapter
        try:
            for offset in range(0, 8, 2):
                pending = [row for row in ledger['calls'][offset:offset + 2] if row['state'] == 'NOT_SENT']
                if not pending:
                    continue
                verify_parent(plan); pricing_fresh(plan, evidence)
                needed = sum((_decimal(row['reserve_usd']) - _decimal(row['reservation_usd']) for row in pending), Decimal(0))
                require(_decimal(accounting(plan, ledger)['combined_committed_usd']) + needed <= Decimal('7.00'),
                        'next complete paired block exceeds combined cap')
                for row in pending:
                    row['reservation_usd'] = row['reserve_usd']
                _save(enabled / 'ledger.json', ledger)
                for row in pending:
                    verify_parent(plan); pricing_fresh(plan, evidence)
                    payload = bound(row['payload']); adapter.validate_payload(row['builder'], payload)
                    row.update(state='SENT_PENDING', send_status='POSSIBLY_SENT', started_at_utc=now())
                    _save(enabled / 'ledger.json', ledger)
                    _dispatch(plan, row, ledger, enabled, adapter)
            ledger['status'] = 'COMPLETE' if all(row['state'] == 'SETTLED' for row in ledger['calls']) else 'PARTIAL_WITH_UNCERTAIN_CALLS'
        except BaseException as exc:
            ledger['status'] = 'SAFE_STOP'
            ledger['halt'] = {'type': type(exc).__name__, 'reason': str(exc) if isinstance(exc, RuntimeErrorSafe) else 'execution interrupted; inspect saved state'}
        finally:
            _save(enabled / 'ledger.json', ledger)
            try:
                reason = 'FINAL' if ledger['status'] == 'COMPLETE' else 'SAFE_STOP'
                if checkpoint_callback:
                    checkpoint = checkpoint_callback(plan_path, enabled / 'ledger.json', reason)
                else:
                    from .checkpoint import create_checkpoint
                    checkpoint = create_checkpoint(plan_path, enabled / 'ledger.json', reason=reason)
                ledger['last_checkpoint'] = checkpoint
            except Exception as exc:
                ledger['checkpoint_failure'] = {'type': type(exc).__name__, 'reason': 'save-only retry required'}
            _save(enabled / 'ledger.json', ledger)
        return _result(plan, ledger)


def _dispatch(plan, row, ledger, enabled, adapter):
    """Exactly one POST; durable raw precedes any reservation settlement."""
    raw, transport_failure = None, None
    try:
        raw = adapter.send_request(row['builder'], bound(row['payload']), api_allowed=True, permission=True,
                                   timeout=plan['timeout_seconds'])
        row.update(send_status='POST_RETURNED', response_received=True, http_status=200, raw_complete=True)
    except Exception as exc:
        transport_failure = type(exc).__name__
        raw = getattr(exc, 'raw_response', None)
        row.update(http_status=getattr(exc, 'http_status', None), response_received=getattr(exc, 'http_response_received', False) is True,
                   failure_kind=getattr(exc, 'failure_kind', 'transport_failure'), raw_complete=getattr(exc, 'raw_complete', False) is True)
    if raw is not None:
        row['raw'] = frozen._raw_save(enabled / 'responses' / (row['call_id'] + '.raw.json'), raw)
        _save(enabled / 'ledger.json', ledger)
    rates = plan['pricing_contract']['rates_per_million'][row['builder']]
    pricing = {'input_usd_per_million': rates['input'], 'output_usd_per_million': rates['output']}
    if transport_failure:
        row.update(state='UNKNOWN_USAGE', usage_known=False, transport_error_type=transport_failure)
        if raw is not None and row['raw_complete']:
            try:
                response = adapter.parse_response(row['builder'], raw)
                usage = adapter.classify_usage(row['builder'], response.usage, pricing)
                if usage['usage_known'] is True:
                    cost = str(_decimal(usage['cost_usd']))
                    row.update(state='SETTLED', usage_known=True, usage=usage, known_cost_usd=cost)
            except Exception:
                pass
        _save(enabled / 'ledger.json', ledger)
        _budget_guard(plan, row, ledger)
        raise RuntimeErrorSafe('external request failure; unresolved reservations retained')
    response = adapter.parse_response(row['builder'], raw)
    row.update(completed_response=response.complete, response_model=response.model, response_id=response.response_id,
               failure_kind=response.failure_kind, model_output_failure=response.failure_kind in MODEL_FAILURES)
    if response.failure_kind in CONTRACT_FAILURES or (response.complete is not True and not row['model_output_failure']):
        ledger['requires_new_explicit_approval'] = True
    _save(enabled / 'ledger.json', ledger)
    if response.text is not None:
        row['response_text_path'] = frozen._raw_save(enabled / 'responses' / (row['call_id'] + '.text.txt'), response.text.encode())
    usage = adapter.classify_usage(row['builder'], response.usage, pricing)
    cost = str(_decimal(usage['cost_usd'])) if usage['usage_known'] is True else None
    row.update(usage_known=usage['usage_known'] is True, usage=usage, finished_at_utc=now())
    if row['usage_known']:
        row.update(state='SETTLED', known_cost_usd=cost)
    else:
        row['state'] = 'UNKNOWN_USAGE'
    _save(enabled / 'ledger.json', ledger)
    require(row['usage_known'], 'unknown usage; full reservation retained')
    _budget_guard(plan, row, ledger)
    require(not ledger.get('requires_new_explicit_approval'), 'response contract failure; new explicit approval required')


def _budget_guard(plan, row, ledger):
    if row['usage_known']:
        require(_decimal(row['known_cost_usd']) <= _decimal(row['reserve_usd']), 'observed usage exceeded full reservation')
    require(_decimal(accounting(plan, ledger)['combined_committed_usd']) <= Decimal('7.00'), 'combined budget exceeded')
