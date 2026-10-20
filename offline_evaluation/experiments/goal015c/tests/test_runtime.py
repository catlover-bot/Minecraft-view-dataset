"""Synthetic-only runtime fixtures. No credentials, old files, or HTTP calls."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import signal
import time
from types import SimpleNamespace

import pytest

from offline_evaluation.experiments.goal015c import adapter as wire_adapter
from offline_evaluation.experiments.goal015c import planning, pricing, runtime


def fixture_plan(tmp_path, *, cap='3.25'):
    tmp_path.mkdir(parents=True,exist_ok=True)
    sources = tmp_path/'old-fixture'; sources.mkdir()
    bindings = {}
    for key in ('parent_plan','parent_ledger','parent_approval','parent_authority',
                'parent_execution_lock','child_plan'):
        path = sources/(key+'.json')
        path.write_text('{}\n')
        bindings[key] = planning.descriptor(path)
    bindings['child_required_absent'] = [str(sources/name) for name in
                                      ('enabled','execution.lock','completion.lock')]
    acquired = dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=2)
    body = tmp_path/'synthetic-document.txt'; body.write_text('SYNTHETIC FIXTURE ONLY\n')
    documents = {role:{'requested_url':url,'final_url':url,'http_status':200,
                'request_started_utc':acquired.isoformat(),'retrieved_utc':acquired.isoformat(),
                'body':planning.descriptor(body)} for role,url in pricing.SOURCES.items()}
    contract = pricing.contract()
    evidence = tmp_path/'fixture-reviewed.json'
    evidence.write_text(json.dumps({'schema':'goal015c_reviewed_evidence_v1','reviewed':True,
        'contract':contract,'contract_sha256':hashlib.sha256(wire_adapter.canonical_json(contract).encode()).hexdigest(),
        'documents':documents,'reviewed_utc':acquired.isoformat(),
        'valid_until_utc':(acquired+dt.timedelta(hours=24)).isoformat()}))
    planning.create_plan(tmp_path/'diagnostic',evidence,bindings,fixture_only=True,
                         proposed_budget_usd=cap)
    return tmp_path/'diagnostic/plan.json'


def fixture_checkpoint(*args):
    return {'fixture_only':True,'status':'MOCK_LOCAL_CHECKPOINT'}


class MockAdapter:
    fixture_only = True
    uses_live_transport = False

    def __init__(self, *, unknown_at=None, transport_at=None, failure_at=None,
                 mismatch_at=None, before=None, text=None, cost=None):
        self.sent = []
        self.unknown_at = unknown_at
        self.transport_at = transport_at
        self.failure_at = failure_at
        self.mismatch_at = mismatch_at
        self.before = before
        self.text = text if text is not None else json.dumps(wire_adapter.EXPECTED_DOCUMENT)
        self.cost = cost

    def send_wire(self,builder,condition,wire,**kwargs):
        number = len(self.sent)+1
        if self.before:
            self.before(number,builder,condition,kwargs)
        assert kwargs['api_allowed'] is True and kwargs['permission'] is True
        assert kwargs['timeout'] == 600
        assert hashlib.sha256(wire).hexdigest() == kwargs['expected_sha256']
        assert len(wire) == kwargs['expected_bytes']
        metadata = {'request_data_equal':True,'request_data_sha256':kwargs['expected_sha256'],
                    'request_data_bytes':len(wire),
                    'endpoint_host':{'openai':'api.openai.com','anthropic':'api.anthropic.com'}[builder]}
        kwargs['before_send'](metadata)
        self.sent.append((builder,condition))
        if number == self.transport_at:
            raise OSError('fixture transport interruption')
        failure = ('response_model_mismatch' if number == self.mismatch_at else
                   'truncation' if number == self.failure_at else None)
        # Valid provider response envelopes allow independent evaluator extraction.
        text = self.text
        if builder == 'openai':
            value = {'id':f'fixture-{number}','model':wire_adapter.MODELS[builder],'status':'completed',
                     'output':[{'type':'message','role':'assistant','status':'completed',
                                'content':[{'type':'output_text','text':text}]}],
                     'usage':{'input_tokens':10,'output_tokens':20}}
        else:
            value = {'id':f'fixture-{number}','model':wire_adapter.MODELS[builder],'type':'message',
                     'role':'assistant','stop_reason':'end_turn','content':[{'type':'text','text':text}],
                     'usage':{'input_tokens':10,'output_tokens':20}}
        value['_fixture_number'] = number
        value['_fixture_failure'] = failure
        if number == self.unknown_at:
            value['usage'] = None
        return {'raw':json.dumps(value).encode(),'http_status':200,'request_id':f'fixture-http-{number}',**metadata}

    def parse_response(self,builder,raw):
        value = json.loads(raw)
        failure = value['_fixture_failure']
        return SimpleNamespace(text=self.text,response_id=value['id'],model=value['model'],
            complete=failure is None,failure_kind=failure,
            usage={'fixture_number':value['_fixture_number']})

    def classify_usage(self,builder,usage,terms):
        if usage['fixture_number'] == self.unknown_at:
            return {'usage_known':False,'cost_usd':None}
        result = wire_adapter.classify_usage(builder,{'input_tokens':10,'output_tokens':20},terms)
        if self.cost is not None:
            result['cost_usd'] = self.cost
        return result


def start(path, fake=None, **kwargs):
    return runtime.execute(path,adapter=fake or MockAdapter(),confirmation=lambda p:True,
                           checkpoint_callback=fixture_checkpoint,**kwargs)


def ledger(path):
    return json.loads((Path(path).parent/'enabled/ledger.json').read_text())


def test_preview_is_read_only_and_full_12(tmp_path):
    path = fixture_plan(tmp_path)
    view = runtime.preview(path)
    assert view['planned_calls'] == 12 and view['maximum_reservation_usd'] == '3.176256'
    assert view['model_account_availability'] == 'NOT_RUN'
    assert runtime.preflight(path,require_unstarted=True)['status'] == 'PREFLIGHT_PASS'
    assert not (path.parent/'enabled').exists() and not (path.parent/'execution.lock').exists()


def test_full12_durable_wire_raw_and_duplicate(tmp_path):
    path = fixture_plan(tmp_path); fake = MockAdapter()
    result = start(path,fake)
    assert result['status'] == 'COMPLETE' and result['calls_settled'] == 12
    assert len(fake.sent) == 12
    for row in ledger(path)['calls']:
        assert row['wire_evidence']['request_data_equal'] is True
        assert planning.descriptor(row['raw']['path']) == row['raw']
        assert row['reservation_usd'] == row['reserve_usd']
    called = []
    with pytest.raises(runtime.RuntimeErrorSafe):
        runtime.execute(path,adapter=fake,confirmation=lambda p:called.append(True))
    assert called == [] and len(fake.sent) == 12
    result = runtime.execute(path,resume=True,adapter=fake,checkpoint_callback=fixture_checkpoint)
    assert result['calls_settled'] == 12 and len(fake.sent) == 12


@pytest.mark.parametrize('mode',['unknown','transport'])
def test_unknown_holds_and_resume_only_not_sent(tmp_path,mode):
    path = fixture_plan(tmp_path); fake = MockAdapter(**{mode+'_at':1})
    first = start(path,fake)
    assert first['status'] == 'SAFE_STOP' and first['calls_uncertain'] == 1
    assert first['budget']['unsettled_and_unknown_holds_usd'] == '0.165536'
    fake.unknown_at = fake.transport_at = None
    second = runtime.execute(path,resume=True,adapter=fake,checkpoint_callback=fixture_checkpoint)
    assert second['calls_settled'] == 11 and second['calls_uncertain'] == 1 and len(fake.sent) == 12
    assert ledger(path)['calls'][0]['state'] == 'UNKNOWN_USAGE'


def test_known_model_failure_continues_but_unknown_contract_does_not(tmp_path):
    path = fixture_plan(tmp_path/'normal')
    assert start(path,MockAdapter(failure_at=1))['calls_settled'] == 12
    path = fixture_plan(tmp_path/'mismatch'); fake = MockAdapter(mismatch_at=1,unknown_at=1)
    assert start(path,fake)['calls_uncertain'] == 1
    with pytest.raises(runtime.RuntimeErrorSafe,match='contract failure'):
        runtime.execute(path,resume=True,adapter=fake)
    assert len(fake.sent) == 1


@pytest.mark.parametrize('text,kind',[('```json\n{}\n```','parse_failure'),('{"other":[]}','validation_failure')])
def test_known_bad_output_continues_all_conditions(tmp_path,text,kind):
    path = fixture_plan(tmp_path)
    result = start(path,MockAdapter(text=text))
    assert result['calls_settled'] == 12 and result['known_model_output_failures'] == 12
    assert all(r['failure_kind'] == kind for r in ledger(path)['calls'])


def test_pricing_expiry_before_approval_and_each_send(tmp_path,monkeypatch):
    path = fixture_plan(tmp_path/'before'); called = []
    original = pricing.now
    monkeypatch.setattr(pricing,'now',lambda:original()+dt.timedelta(days=2))
    with pytest.raises(pricing.EvidenceError,match='expired'):
        runtime.execute(path,adapter=MockAdapter(),confirmation=lambda p:called.append(True))
    assert called == [] and not (path.parent/'execution.lock').exists()
    monkeypatch.setattr(pricing,'now',original)
    path = fixture_plan(tmp_path/'during')
    def before(number,*args):
        if number == 1:
            monkeypatch.setattr(pricing,'now',lambda:original()+dt.timedelta(days=2))
    fake = MockAdapter(before=before)
    result = start(path,fake)
    assert result['calls_settled'] == 1 and result['calls_not_sent'] == 11 and len(fake.sent) == 1


def test_orphan_response_not_sent_refuses_resume(tmp_path):
    path = fixture_plan(tmp_path,cap='0.2')
    start(path)
    row = ledger(path)['calls'][1]
    (path.parent/'enabled/responses'/(row['call_id']+'.raw.json')).write_bytes(b'{}')
    with pytest.raises(runtime.RuntimeErrorSafe,match='orphan'):
        runtime.execute(path,resume=True,adapter=MockAdapter())


def test_whole_call_reservation_and_separate_budget(tmp_path):
    path = fixture_plan(tmp_path,cap='0.2'); fake = MockAdapter()
    result = start(path,fake)
    assert result['calls_settled'] == 1 and result['calls_not_sent'] == 11
    assert result['budget']['old_budget_credit_used_usd'] == '0'
    assert runtime._decimal(result['budget']['known_cost_usd']) == runtime._decimal('0.0000425')
    assert len(fake.sent) == 1


def test_parent_lock_and_parent_change_during_approval(tmp_path):
    path = fixture_plan(tmp_path); plan = runtime.verify_plan(path)
    with open(plan['source_bindings']['parent_execution_lock']['path'],'rb') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(runtime.RuntimeErrorSafe,match='parent execution'):
            start(path)
    def changed(p):
        Path(p['source_bindings']['parent_ledger']['path']).write_text('{"changed":true}')
        return True
    with pytest.raises(planning.PlanError,match='bound file changed'):
        runtime.execute(path,adapter=MockAdapter(),confirmation=changed)
    assert not (path.parent/'enabled').exists()


def test_production_hooks_and_fixture_missing_adapter_refused(tmp_path,monkeypatch):
    path = fixture_plan(tmp_path)
    with pytest.raises(runtime.RuntimeErrorSafe,match='explicit fake adapter'):
        runtime.execute(path,confirmation=lambda p:True)
    with pytest.raises(runtime.RuntimeErrorSafe,match='disable live transport'):
        runtime.execute(path,adapter=wire_adapter,confirmation=lambda p:True)
    plan = runtime.verify_plan(path); plan.pop('fixture_only')
    monkeypatch.setattr(runtime,'verify_plan',lambda p:plan)
    with pytest.raises(runtime.RuntimeErrorSafe,match='test hooks forbidden'):
        runtime.execute(path,adapter=MockAdapter(),confirmation=lambda p:True)


def test_wire_tamper_copy_scope_and_authority(tmp_path):
    path = fixture_plan(tmp_path/'a'); plan = runtime.verify_plan(path)
    wire = Path(plan['schedule'][0]['wire']['path']); wire.write_bytes(wire.read_bytes()+b' ')
    with pytest.raises(planning.PlanError,match='bound file changed'):
        start(path)
    path = fixture_plan(tmp_path/'b')
    copy = tmp_path/'plan.json'; copy.write_bytes(path.read_bytes())
    with pytest.raises(planning.PlanError,match='copied'):
        runtime.preview(copy)
    plan = json.loads(path.read_text()); plan['schedule'].pop()
    plan['plan_id'] = planning.plan_identity(plan); path.write_text(json.dumps(plan))
    with pytest.raises(planning.PlanError,match='twelve'):
        runtime.preview(path)
    path = fixture_plan(tmp_path/'c'); start(path)
    authority = path.parent/'enabled/authority.json'
    value = json.loads(authority.read_text()); value['scope_calls'] = 13; authority.write_text(json.dumps(value))
    with pytest.raises(runtime.RuntimeErrorSafe,match='authority'):
        runtime.status(path)


def test_raw_durable_before_settlement(tmp_path):
    path = fixture_plan(tmp_path)
    class Inspect(MockAdapter):
        def classify_usage(self,builder,usage,terms):
            row = ledger(path)['calls'][usage['fixture_number']-1]
            assert row['state'] == 'SENT_PENDING' and row['known_cost_usd'] is None
            assert planning.descriptor(row['raw']['path']) == row['raw']
            assert row['wire_evidence']['request_data_equal'] is True
            return super().classify_usage(builder,usage,terms)
    assert start(path,Inspect())['calls_settled'] == 12


def _pty_call(path, answer, *, execute=False):
    pid,master = pty.fork()
    if pid == 0:
        try:
            if execute:
                result = runtime.execute(path,adapter=MockAdapter(),checkpoint_callback=fixture_checkpoint)
                assert result['calls_settled'] == 12
                try:
                    runtime.execute(path,adapter=MockAdapter(),checkpoint_callback=fixture_checkpoint)
                except runtime.RuntimeErrorSafe:
                    pass
                else:
                    os._exit(4)
            else:
                runtime.confirm_tty(runtime.verify_plan(path))
            os._exit(0)
        except BaseException:
            os._exit(3)
    output = b''; deadline = time.monotonic()+10
    try:
        while b'\n> ' not in output and time.monotonic()<deadline:
            if select.select([master],[],[],0.1)[0]:
                output += os.read(master,65536)
        assert b'Type exactly:' in output
        os.write(master,answer)
        while time.monotonic()<deadline:
            ended,status = os.waitpid(pid,os.WNOHANG)
            if ended:
                return os.waitstatus_to_exitcode(status),output
            if select.select([master],[],[],0.1)[0]:
                try:
                    output += os.read(master,65536)
                except OSError:
                    pass
        raise AssertionError('fixture PTY timed out')
    finally:
        try: os.kill(pid,signal.SIGKILL)
        except ProcessLookupError: pass
        try: os.waitpid(pid,0)
        except ChildProcessError: pass
        os.close(master)


@pytest.mark.parametrize('answer',[b'\n',b'WRONG\n',b'\x04'])
def test_real_pty_rejects_enter_wrong_and_eof(tmp_path,answer):
    path = fixture_plan(tmp_path)
    assert _pty_call(path,answer)[0] == 3
    assert not (path.parent/'enabled').exists()


def test_real_pty_exact_approval_and_duplicate_execution(tmp_path):
    path = fixture_plan(tmp_path)
    answer = (runtime.approval_text(runtime.verify_plan(path))+'\n').encode()
    code,output = _pty_call(path,answer,execute=True)
    assert code == 0 and output.count(b'Type exactly:') == 1
    assert len(ledger(path)['calls']) == 12


def test_non_tty_read_failure(tmp_path,monkeypatch):
    path = fixture_plan(tmp_path)
    monkeypatch.setattr('builtins.open',lambda *a,**k: (_ for _ in ()).throw(OSError()))
    with pytest.raises(runtime.RuntimeErrorSafe,match='TTY unavailable'):
        runtime.confirm_tty(json.loads(path.read_text()))
