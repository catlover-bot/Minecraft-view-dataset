"""Goal015C standalone twelve-call lifecycle; never starts either old experiment.

All files are new canonical diagnostic outputs. Frozen helpers provide durable
files only; this module never calls a parent runtime's execute function.
"""
from contextlib import contextmanager, ExitStack
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ..goal015 import runtime as files

RuntimeErrorSafe = files.RuntimeErrorSafe
require, descriptor, read_json, path_safe = files.require, files.descriptor, files.read_json, files.path_safe
canonical, identity, bound, _save, _decimal, now = files.canonical, files.identity, files.bound, files._save, files._decimal, files.now
MODEL_FAILURES = frozenset(('parse_failure', 'validation_failure', 'execution_failure', 'truncation', 'refusal'))
CONTRACT_FAILURES = frozenset(('response_model_mismatch', 'unexpected_response_feature', 'incomplete_response', 'wire_mismatch'))
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_BINDING = REPOSITORY_ROOT/'reports/server_only/goal015c/EXECUTION_BINDING.json'


def verify_plan(plan_path):
    from .planning import verify_plan as verify
    return verify(plan_path)


def _pricing(plan, evidence=None):
    from .pricing import verify_evidence
    verify_evidence(plan, evidence)
    ref = plan['pricing_evidence'] if evidence is None else evidence if isinstance(evidence, dict) else descriptor(evidence)
    require(descriptor(ref['path']) == ref, 'pricing evidence changed')
    return ref


def metadata_binding(plan):
    """One digest binds scope/cap/wire/code/source/pricing to the new authority."""
    selected = {key:plan[key] for key in ('plan_id','canonical_root','proposed_budget_usd','schedule',
                                         'pricing_contract','implementation_hashes','source_bindings')}
    selected['specification'] = plan.get('specification')
    return hashlib.sha256(canonical(selected)).hexdigest()


def approval_text(plan):
    return f"APPROVE GOAL015C {plan['plan_id']} NEW BUDGET USD {plan['proposed_budget_usd']} MAX 12 CALLS RETRY 0"


def confirm_tty(plan):
    expected = approval_text(plan)
    try:
        with open('/dev/tty','r',encoding='utf-8') as reader, open('/dev/tty','w',encoding='utf-8') as writer:
            require(os.isatty(reader.fileno()) and os.isatty(writer.fileno()), 'real controlling TTY required')
            writer.write('Goal015C new standalone budget USD '+plan['proposed_budget_usd']+
                         '; maximum12 diagnostic calls. Prior USD4.43765700 stays separate; old USD5/7 authority is not reused.\n'
                         +'Type exactly: '+expected+'\n> ')
            writer.flush()
            answer = reader.readline(len(expected)+4)
            require(answer != '' and answer.rstrip('\r\n') == expected, 'exact new diagnostic approval absent')
    except RuntimeErrorSafe:
        raise
    except (OSError,UnicodeError,ValueError):
        raise RuntimeErrorSafe('TTY unavailable or read/write failed') from None
    return True


@contextmanager
def locks(plan, *, create=False):
    """Parent lock first; old files opened rb; diagnostic lock created only after approval."""
    bindings = plan['source_bindings']
    with ExitStack() as stack:
        parent = stack.enter_context(path_safe(bindings['parent_execution_lock']['path']).open('rb'))
        try:
            fcntl.flock(parent.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeErrorSafe('parent execution already active') from None
        stack.callback(fcntl.flock,parent.fileno(),fcntl.LOCK_UN)
        for name in bindings['child_required_absent']:
            require(not os.path.lexists(name), 'old child execution state appeared; stop diagnostic')
        root = path_safe(plan['canonical_root']); lock_path = path_safe(root/'execution.lock')
        if create:
            completion = root/'completion.lock'
            if completion.exists():
                guard = stack.enter_context(path_safe(completion).open('rb'))
                try:
                    fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeErrorSafe('diagnostic finish/save already active') from None
                stack.callback(fcntl.flock,guard.fileno(),fcntl.LOCK_UN)
            stack.enter_context(files.canonical_lock(root))
            files._fsync_directory(root)
        elif lock_path.exists():
            current = stack.enter_context(lock_path.open('rb'))
            try:
                fcntl.flock(current.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeErrorSafe('diagnostic execution already active') from None
            stack.callback(fcntl.flock,current.fileno(),fcntl.LOCK_UN)
        yield


def _unstarted(plan):
    root = Path(plan['canonical_root'])
    for relative in ('enabled','approval.json','authority.json','ledger.json','execution.lock'):
        require(not os.path.lexists(root/relative), 'existing diagnostic execution; review status and explicitly resume or finish')


def _committed_registry(plan):
    if plan.get('fixture_only') is True:
        return
    try:
        committed = subprocess.check_output(['git','show','HEAD:reports/server_only/goal015c/EXECUTION_BINDING.json'],
                                            cwd=REPOSITORY_ROOT,stderr=subprocess.DEVNULL)
        branch = subprocess.check_output(['git','branch','--show-current'],cwd=REPOSITORY_ROOT).decode().strip()
    except (OSError,subprocess.CalledProcessError):
        raise RuntimeErrorSafe('committed diagnostic registry required') from None
    require(committed == EXECUTION_BINDING.read_bytes() and branch == 'research/server-only-20261013',
            'committed canonical diagnostic registry on dedicated branch required')


def preflight(plan_path, *, pricing_evidence=None, require_unstarted=False):
    plan = verify_plan(plan_path)
    _committed_registry(plan)
    with locks(plan):
        require(verify_plan(plan_path) == plan, 'plan or old source state changed')
        evidence = _pricing(plan,pricing_evidence)
        if require_unstarted:
            _unstarted(plan)
    return {'status':'PREFLIGHT_PASS','plan_id':plan['plan_id'],'pricing_evidence':evidence,
            'parent_child_exclusion_checked':True,'required_unstarted':require_unstarted,
            'api_allowed':False,'budget_approved':False,'research_api_requests_sent':0}


def accounting(ledger):
    known = sum((_decimal(row['known_cost_usd']) for row in ledger['calls'] if row['known_cost_usd'] is not None),Decimal(0))
    held = sum((_decimal(row['reservation_usd']) for row in ledger['calls'] if row['state'] != 'SETTLED'),Decimal(0))
    return {'known_cost_usd':str(known),'unsettled_and_unknown_holds_usd':str(held),'committed_usd':str(known+held),
            'old_parent_cost_usd_separate':'4.43765700','old_budget_credit_used_usd':'0'}


def _validate_ledger(plan, ledger, approval, *, for_resume=False, verify_files=True):
    require(ledger['plan_id'] == plan['plan_id'] == approval['plan_id']
            and approval['canonical_root'] == plan['canonical_root'] and approval['status'] == 'APPROVED'
            and approval['scope_calls'] == 12 and approval['approved_budget_usd'] == plan['proposed_budget_usd']
            and approval['approval_text'] == approval_text(plan)
            and approval['metadata_binding_sha256'] == metadata_binding(plan), 'diagnostic approval binding differs')
    require(len(ledger['calls']) == len(plan['schedule']) == 12, 'exact twelve-call ledger required')
    if for_resume:
        require(not ledger.get('requires_new_explicit_approval') and not any(
            row.get('failure_kind') in CONTRACT_FAILURES for row in ledger['calls']), 'contract failure blocks same-plan continuation')
    responses = Path(plan['canonical_root'])/'enabled/responses'
    for original,row in zip(plan['schedule'],ledger['calls']):
        require(all(row.get(key) == value for key,value in original.items()), 'diagnostic schedule changed')
        require(row['state'] in ('NOT_SENT','SENT_PENDING','UNKNOWN_USAGE','SETTLED'), 'invalid diagnostic state')
        reserve = _decimal(row['reservation_usd']); ceiling = _decimal(row['reserve_usd'])
        require(reserve in (Decimal(0),ceiling), 'reservation changed')
        if row['state'] == 'NOT_SENT':
            require(row['send_status'] == 'NOT_SENT' and row['raw'] is None and row['response_text_path'] is None
                    and row['response_received'] is False and row['completed_response'] is False
                    and row.get('wire_evidence') is None and 'started_at_utc' not in row and 'finished_at_utc' not in row,
                    'NOT_SENT has transmission evidence')
            if verify_files:
                require(not any(os.path.lexists(responses/(row['call_id']+suffix)) for suffix in ('.raw.json','.text.txt')),
                        'NOT_SENT has orphan response file; cannot resend')
        else:
            require(reserve == ceiling, 'attempted call needs full reservation')
            wire = row.get('wire_evidence')
            require(isinstance(wire,dict) and wire['presend_sha256'] == row['wire']['sha256']
                    and wire['presend_bytes'] == row['wire']['bytes'], 'attempted wire identity differs')
        if row['state'] == 'SETTLED':
            require(row['usage_known'] is True and row['known_cost_usd'] is not None and row['raw'] is not None,
                    'settlement needs known usage and durable raw')
            cost = _decimal(row['known_cost_usd'])
            if for_resume:
                require(cost <= ceiling, 'observed cost exceeded reservation')
        else:
            require(row['usage_known'] is False and row['known_cost_usd'] is None, 'unknown holds cannot be released')
        for key,suffix in (('raw','.raw.json'),('response_text_path','.text.txt')):
            if row.get(key) is not None:
                require(Path(row[key]['path']) == responses/(row['call_id']+suffix), 'response path differs from canonical call')
                if verify_files:
                    require(descriptor(row[key]['path']) == row[key], 'saved response evidence changed')
    if for_resume:
        require(_decimal(accounting(ledger)['committed_usd']) <= _decimal(plan['proposed_budget_usd']), 'standalone diagnostic cap exceeded')


def verify_enabled(plan, *, for_resume=False):
    enabled = Path(plan['canonical_root'])/'enabled'
    approval,authority,ledger = (read_json(enabled/name) for name in ('approval.json','authority.json','ledger.json'))
    require(authority.get('status') == 'APPROVED' and authority.get('plan_id') == plan['plan_id']
            and authority.get('canonical_root') == plan['canonical_root'] and authority.get('scope_calls') == 12
            and authority.get('max_retries') == 0 and authority.get('metadata_binding_sha256') == metadata_binding(plan)
            and authority.get('plan') == descriptor(enabled.parent/'plan.json')
            and authority.get('approval') == descriptor(enabled/'approval.json'), 'diagnostic authority binding changed')
    _validate_ledger(plan,ledger,approval,for_resume=for_resume)
    return {'approval':approval,'authority':authority,'ledger':ledger}


def preview(plan_path):
    plan = verify_plan(plan_path)
    return {'status':'READY_AWAITING_EXPLICIT_APPROVAL','plan_id':plan['plan_id'],'planned_calls':12,
            'proposed_new_budget_usd':plan['proposed_budget_usd'],'maximum_reservation_usd':str(sum((_decimal(r['reserve_usd']) for r in plan['schedule']),Decimal(0))),
            'prior_parent_cost_usd_separate':'4.43765700','budget_approved':False,'api_allowed':False,
            'model_account_availability':'NOT_RUN','concurrency':1,'max_retries':0,
            'approval_text':approval_text(plan),'research_api_requests_sent':0}


def _result(plan,ledger):
    return {'status':ledger['status'],'plan_id':plan['plan_id'],'planned_calls':12,'budget':accounting(ledger),
            'calls_settled':sum(r['state']=='SETTLED' for r in ledger['calls']),
            'calls_not_sent':sum(r['state']=='NOT_SENT' for r in ledger['calls']),
            'calls_uncertain':sum(r['state'] in ('SENT_PENDING','UNKNOWN_USAGE') for r in ledger['calls']),
            'known_model_output_failures':sum(r.get('model_output_failure') is True for r in ledger['calls']),
            'checkpoint':ledger.get('last_checkpoint'),'failure':ledger.get('halt'),
            'checkpoint_failure':ledger.get('checkpoint_failure'),'fixture_only':plan.get('fixture_only') is True,
            'settlement_does_not_imply_task_pass':True}


def status(plan_path):
    plan = verify_plan(plan_path)
    with locks(plan):
        if not (Path(plan['canonical_root'])/'enabled').exists():
            return preview(plan_path)
        return _result(plan,verify_enabled(plan)['ledger'])


def _progress(plan,ledger,row,number,phase):
    print(json.dumps({'event':'goal015c_call_progress','phase':phase,'call':number,'planned':12,
        'call_id':row['call_id'],'condition':row['condition'],'builder':row['builder'],'state':row['state'],
        'known_cost_usd':accounting(ledger)['known_cost_usd'],
        'hold_usd':accounting(ledger)['unsettled_and_unknown_holds_usd']},sort_keys=True),flush=True)


def _dispatch(plan,ledger,row,enabled,adapter):
    wire = path_safe(row['wire']['path']).read_bytes()
    require(len(wire) == row['wire']['bytes'] and hashlib.sha256(wire).hexdigest() == row['wire']['sha256'], 'wire bytes changed')
    row.update(state='SENT_PENDING',send_status='POSSIBLY_SENT',started_at_utc=now(),
               wire_evidence={'presend_sha256':hashlib.sha256(wire).hexdigest(),'presend_bytes':len(wire),
                              'request_data_equal':None,'request_data_sha256':None,'request_data_bytes':None})
    _save(enabled/'ledger.json',ledger)
    called = False
    def before_send(metadata):
        nonlocal called
        require(not called,'multiple POST dispatches forbidden'); called = True
        require(metadata.get('request_data_equal') is True
                and metadata.get('request_data_sha256') == row['wire']['sha256']
                and metadata.get('request_data_bytes') == row['wire']['bytes'], 'Request.data does not equal frozen wire')
        allowed = {'openai':'api.openai.com','anthropic':'api.anthropic.com'}[row['builder']]
        require(metadata.get('endpoint_host') == allowed, 'unapproved endpoint host')
        row['wire_evidence'].update({key:metadata[key] for key in
            ('request_data_equal','request_data_sha256','request_data_bytes','endpoint_host')})
        row['wire_evidence']['checked_at_utc'] = now()
        _save(enabled/'ledger.json',ledger)
    raw = None; transport_error = None
    try:
        result = adapter.send_wire(row['builder'],row['condition'],wire,
            expected_sha256=row['wire']['sha256'],expected_bytes=row['wire']['bytes'],
            api_allowed=True,permission=True,timeout=plan['timeout_seconds'],before_send=before_send)
        require(called,'transport did not certify Request.data before dispatch')
        raw = result['raw']
        require(type(raw) is bytes,'transport raw bytes required')
        row.update(send_status='POST_RETURNED',response_received=True,http_status=result.get('http_status'),
                   request_id=result.get('request_id'),raw_complete=True)
    except Exception as exc:
        transport_error = type(exc).__name__; raw = getattr(exc,'raw_response',None)
        row.update(http_status=getattr(exc,'http_status',getattr(exc,'status_code',None)),
            request_id=getattr(exc,'request_id',None),response_received=getattr(exc,'http_response_received',False) is True,
            raw_complete=getattr(exc,'raw_complete',False) is True,
            failure_kind=getattr(exc,'failure_kind',getattr(exc,'kind','transport_failure')))
        if isinstance(exc,RuntimeErrorSafe):
            row['failure_kind'] = 'wire_mismatch'
        if row['failure_kind'] in CONTRACT_FAILURES:
            ledger['requires_new_explicit_approval'] = True
    if raw is not None:
        row['raw'] = files._raw_save(enabled/'responses'/(row['call_id']+'.raw.json'),raw)
        _save(enabled/'ledger.json',ledger)
    if transport_error:
        row.update(state='UNKNOWN_USAGE',usage_known=False,transport_error_type=transport_error)
        _save(enabled/'ledger.json',ledger)
        if raw is not None and row['raw_complete']:
            try:
                response = adapter.parse_response(row['builder'],raw)
                if response.failure_kind in CONTRACT_FAILURES:
                    row['failure_kind'] = response.failure_kind
                    ledger['requires_new_explicit_approval'] = True
                usage = adapter.classify_usage(row['builder'],response.usage,plan['pricing_contract']['providers'][row['builder']])
                if usage['usage_known'] is True:
                    cost = _decimal(usage['cost_usd'])
                    row.update(state='SETTLED',usage_known=True,usage=usage,known_cost_usd=str(cost))
                    _save(enabled/'ledger.json',ledger)
            except Exception:
                pass
        if row['usage_known']:
            require(_decimal(row['known_cost_usd']) <= _decimal(row['reserve_usd']),
                    'observed usage exceeded reservation')
            require(_decimal(accounting(ledger)['committed_usd']) <= _decimal(plan['proposed_budget_usd']),
                    'standalone budget exceeded')
        raise RuntimeErrorSafe('external request failure; inspect durable raw and retained holds')
    response = adapter.parse_response(row['builder'],raw)
    row.update(completed_response=response.complete,response_model=response.model,response_id=response.response_id,
               failure_kind=response.failure_kind,model_output_failure=response.failure_kind in MODEL_FAILURES)
    contract_failure = response.failure_kind in CONTRACT_FAILURES or (response.complete is not True and not row['model_output_failure'])
    if contract_failure:
        ledger['requires_new_explicit_approval'] = True
    if response.complete is True and response.failure_kind is None:
        from offline_evaluation.server_v2.core import strict_json, ValidationError
        from offline_evaluation.resmoke_v2.schema import validate_document
        try:
            document = strict_json(response.text)
        except (ValidationError,TypeError,ValueError):
            row.update(failure_kind='parse_failure',model_output_failure=True)
        else:
            try:
                validate_document(document)
            except ValidationError:
                row.update(failure_kind='validation_failure',model_output_failure=True)
    _save(enabled/'ledger.json',ledger)
    if response.text is not None:
        row['response_text_path'] = files._raw_save(enabled/'responses'/(row['call_id']+'.text.txt'),response.text.encode('utf-8'))
    usage = adapter.classify_usage(row['builder'],response.usage,plan['pricing_contract']['providers'][row['builder']])
    if usage['usage_known'] is True:
        cost = _decimal(usage['cost_usd'])
        row.update(state='SETTLED',usage_known=True,usage=usage,known_cost_usd=str(cost))
    else:
        row.update(state='UNKNOWN_USAGE',usage_known=False,usage=usage)
    row['finished_at_utc'] = now(); _save(enabled/'ledger.json',ledger)
    require(row['usage_known'],'unknown usage; full reservation retained')
    require(_decimal(row['known_cost_usd']) <= _decimal(row['reserve_usd']),'observed usage exceeded reservation')
    require(_decimal(accounting(ledger)['committed_usd']) <= _decimal(plan['proposed_budget_usd']),'standalone budget exceeded')
    require(not contract_failure,'response contract mismatch; same-plan continuation blocked')


def execute(plan_path, *, pricing_evidence=None, resume=False, adapter=None, confirmation=None, checkpoint_callback=None):
    plan = verify_plan(plan_path); fixture = plan.get('fixture_only') is True
    require(fixture or (adapter is None and confirmation is None and checkpoint_callback is None), 'test hooks forbidden for production plan')
    require(not fixture or adapter is not None,'fixture execution requires an explicit fake adapter')
    require(not fixture or (getattr(adapter,'fixture_only',None) is True
                            and getattr(adapter,'uses_live_transport',None) is False),
            'fixture adapter must explicitly disable live transport')
    preflight(plan_path,pricing_evidence=pricing_evidence,require_unstarted=not resume)
    if not resume:
        require((confirmation or confirm_tty)(plan) is True,'diagnostic approval refused')
    root = path_safe(plan['canonical_root']); enabled = root/'enabled'
    with locks(plan,create=True):
        require(verify_plan(plan_path) == plan,'plan or old state changed while awaiting approval')
        evidence = _pricing(plan,pricing_evidence)
        if resume:
            controls = verify_enabled(plan,for_resume=True); ledger = controls['ledger']
        else:
            require(not os.path.lexists(enabled),'duplicate diagnostic activation refused')
            enabled.mkdir(); files._fsync_directory(root)
            (enabled/'responses').mkdir(); files._fsync_directory(enabled)
            approval = {'status':'APPROVED','plan_id':plan['plan_id'],'canonical_root':str(root),'scope_calls':12,
                'approved_budget_usd':plan['proposed_budget_usd'],'approval_text':approval_text(plan),
                'metadata_binding_sha256':metadata_binding(plan),'approved_at_utc':now(),
                'approval_method':'FIXTURE_ONLY' if fixture else 'EXACT_REAL_TTY'}
            _save(enabled/'approval.json',approval,exclusive=True)
            _save(enabled/'authority.json',{'status':'APPROVED','plan_id':plan['plan_id'],'canonical_root':str(root),
                'scope_calls':12,'max_retries':0,'metadata_binding_sha256':metadata_binding(plan),
                'plan':descriptor(plan_path),'approval':descriptor(enabled/'approval.json')},exclusive=True)
            ledger = {'schema':'goal015c_execution_ledger_v1','plan_id':plan['plan_id'],'status':'APPROVED_NOT_SENT',
                'calls':[dict(row,state='NOT_SENT',reservation_usd='0',known_cost_usd=None,usage_known=False,
                    send_status='NOT_SENT',response_received=False,completed_response=False,model_output_failure=False,
                    failure_kind=None,raw=None,response_text_path=None,wire_evidence=None) for row in plan['schedule']]}
            _save(enabled/'ledger.json',ledger,exclusive=True)
        ledger['accepted_pricing_evidence'] = evidence
        ledger.pop('halt',None)
        ledger.pop('checkpoint_failure',None)
        ledger['status'] = 'RUNNING'; _save(enabled/'ledger.json',ledger)
        if adapter is None:
            from . import adapter
        failure = None
        try:
            for number,row in enumerate(ledger['calls'],1):
                if row['state'] != 'NOT_SENT':
                    continue
                require(verify_plan(plan_path) == plan,'plan/parent/child state changed before dispatch')
                _pricing(plan,pricing_evidence)
                needed = _decimal(row['reserve_usd'])-_decimal(row['reservation_usd'])
                require(_decimal(accounting(ledger)['committed_usd'])+needed <= _decimal(plan['proposed_budget_usd']),
                        'next full call reservation exceeds standalone cap')
                row['reservation_usd'] = row['reserve_usd']; _save(enabled/'ledger.json',ledger)
                _progress(plan,ledger,row,number,'before')
                try:
                    _dispatch(plan,ledger,row,enabled,adapter)
                finally:
                    _progress(plan,ledger,row,number,'after')
            ledger['status'] = 'COMPLETE' if all(r['state']=='SETTLED' for r in ledger['calls']) else 'PARTIAL_WITH_UNCERTAIN_CALLS'
        except BaseException as exc:
            failure = {'type':type(exc).__name__,'reason':str(exc) if isinstance(exc,RuntimeErrorSafe) else 'diagnostic interrupted; inspect durable state'}
            ledger.update(status='SAFE_STOP',halt=failure)
        finally:
            _save(enabled/'ledger.json',ledger)
            try:
                if checkpoint_callback:
                    saved = checkpoint_callback(plan_path,enabled/'ledger.json','SAFE_STOP' if failure else 'FINAL')
                else:
                    from .checkpoint import create_checkpoint
                    saved = create_checkpoint(plan_path,enabled/'ledger.json',reason='SAFE_STOP' if failure else 'FINAL')
                ledger['last_checkpoint'] = saved; _save(enabled/'ledger.json',ledger)
            except Exception as exc:
                ledger['checkpoint_failure'] = {'type':type(exc).__name__,'reason':'local checkpoint failed; retry save only'}
                _save(enabled/'ledger.json',ledger)
        return _result(plan,ledger)
