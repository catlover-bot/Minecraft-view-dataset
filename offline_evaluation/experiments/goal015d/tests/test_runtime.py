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

from offline_evaluation.experiments.goal015d import adapter as wire_adapter
from offline_evaluation.experiments.goal015d import planning, pricing, runtime


FIXTURE_DOCUMENT = {'operations':[
    {'op':'fill','x1':0,'y1':0,'z1':0,'x2':1,'y2':1,'z2':1,'block':'stone'},
    {'op':'carve','x1':0,'y1':0,'z1':0,'x2':0,'y2':0,'z2':0},
    {'op':'set','x':3,'y':0,'z':0,'block':'glass'}]}


def build_fixture(tmp_path, *, cap='5.00'):
    tmp_path.mkdir(parents=True,exist_ok=True)
    sources = tmp_path/'old-fixture'; sources.mkdir()
    bindings = {}
    for key in ('parent_plan','parent_ledger','parent_approval','parent_authority',
                'parent_execution_lock','child_plan','mini_plan','mini_ledger','mini_approval',
                'mini_authority','mini_execution_lock','mini_completion_lock','mini_completion_progress',
                'mini_evaluation','mini_summary','mini_verification'):
        path = sources/(key+'.json')
        path.write_text('{}\n')
        bindings[key] = planning.descriptor(path)
    bindings['child_required_absent'] = [str(sources/name) for name in
                                      ('enabled','execution.lock','completion.lock')]
    acquired = dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=2)
    body = tmp_path/'synthetic-document.txt'; body.write_text('SYNTHETIC FIXTURE ONLY\n')
    from offline_evaluation.experiments.goal015c import pricing as original_pricing
    documents = {role:{'requested_url':url,'final_url':url,'http_status':200,
                'request_started_utc':acquired.isoformat(),'retrieved_utc':acquired.isoformat(),
                'body':planning.descriptor(body)} for role,url in original_pricing.SOURCES.items()}
    contract = original_pricing.contract()
    evidence = tmp_path/'fixture-reviewed.json'
    evidence.write_text(json.dumps({'schema':'goal015c_reviewed_evidence_v1','reviewed':True,
        'contract':contract,'contract_sha256':hashlib.sha256(wire_adapter.canonical_json(contract).encode()).hexdigest(),
        'documents':documents,'reviewed_utc':acquired.isoformat(),
        'valid_until_utc':(acquired+dt.timedelta(hours=24)).isoformat()}))
    import numpy as np
    from offline_evaluation.server_v2 import core,data
    from offline_evaluation.experiments.goal015.evaluate import frozen_grid_sha256
    inputs,gt_sources = {},{}
    input_directory = tmp_path/'diagnostic/inputs';input_directory.mkdir(parents=True)
    for scene_id in ('buildings_100_v4_building_082','buildings_100_v4_building_003'):
        text = json.dumps({'summary':'Synthetic FULL input fixture '+scene_id,
            'building_type':'fixture','shape':'cube','dimensions_estimate':{},
            'materials':[],'elements':[],'rebuild_hints':[],'uncertainties':[]},separators=(',',':'),sort_keys=True)
        source = input_directory/(scene_id+'.input.json');source.write_text(text)
        inputs[scene_id] = {'input':planning.descriptor(source)}
        directory = tmp_path/'buildings_100_v4'/scene_id.removeprefix('buildings_100_v4_')
        gt = directory/'gt';gt.mkdir(parents=True)
        array = np.full((2,4,2),'air',dtype='<U5')
        for (x,y,z),material in core.execute(FIXTURE_DOCUMENT['operations']).voxels.items():
            array[y,x,z] = material
        np.save(gt/'voxels.npy',array,allow_pickle=False)
        (gt/'bbox.json').write_text(json.dumps({'xmin':0,'xmax':3,'ymin':0,'ymax':1,'zmin':0,'zmax':1}))
        gt_sources[scene_id] = {'building_directory':str(directory),
            'voxels_sha256':planning.descriptor(gt/'voxels.npy')['sha256'],
            'bbox_sha256':planning.descriptor(gt/'bbox.json')['sha256'],
            'grid_sha256':frozen_grid_sha256(data.load_scene(directory).grid)}
    planning.create_plan(tmp_path/'diagnostic',evidence,bindings,inputs=inputs,gt_sources=gt_sources,
                         fixture_only=True,proposed_budget_usd=cap)
    return tmp_path/'diagnostic/plan.json'


fixture_plan = build_fixture


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
        self.text = text if text is not None else json.dumps(FIXTURE_DOCUMENT)
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


def test_preview_is_read_only_and_full_16(tmp_path):
    path = fixture_plan(tmp_path)
    view = runtime.preview(path)
    assert view['planned_calls'] == 16 and view['maximum_reservation_usd'] == '4.235008'
    assert view['mini_known_cost_usd_separate'] == '0.01865750'
    assert view['model_account_availability'] == 'NOT_RUN'
    assert runtime.preflight(path,require_unstarted=True)['status'] == 'PREFLIGHT_PASS'
    assert not (path.parent/'enabled').exists() and not (path.parent/'execution.lock').exists()


def test_full16_durable_wire_raw_and_duplicate(tmp_path):
    path = fixture_plan(tmp_path); fake = MockAdapter()
    result = start(path,fake)
    assert result['status'] == 'COMPLETE' and result['calls_settled'] == 16
    assert len(fake.sent) == 16
    for row in ledger(path)['calls']:
        assert row['wire_evidence']['request_data_equal'] is True
        assert planning.descriptor(row['raw']['path']) == row['raw']
        assert row['reservation_usd'] == row['reserve_usd']
    called = []
    with pytest.raises(runtime.RuntimeErrorSafe):
        runtime.execute(path,adapter=fake,confirmation=lambda p:called.append(True))
    assert called == [] and len(fake.sent) == 16
    result = runtime.execute(path,resume=True,adapter=fake,checkpoint_callback=fixture_checkpoint)
    assert result['calls_settled'] == 16 and len(fake.sent) == 16


@pytest.mark.parametrize('mode',['unknown','transport'])
def test_unknown_holds_and_resume_only_not_sent(tmp_path,mode):
    path = fixture_plan(tmp_path); fake = MockAdapter(**{mode+'_at':1})
    first = start(path,fake)
    assert first['status'] == 'SAFE_STOP' and first['calls_uncertain'] == 1
    assert first['budget']['unsettled_and_unknown_holds_usd'] == '0.331072'
    assert ledger(path)['calls'][1]['state'] == 'NOT_SENT'
    assert ledger(path)['calls'][1]['reservation_usd'] == '0.165536'
    fake.unknown_at = fake.transport_at = None
    second = runtime.execute(path,resume=True,adapter=fake,checkpoint_callback=fixture_checkpoint)
    assert second['calls_settled'] == 15 and second['calls_uncertain'] == 1 and len(fake.sent) == 16
    assert ledger(path)['calls'][0]['state'] == 'UNKNOWN_USAGE'


def test_known_model_failure_continues_but_unknown_contract_does_not(tmp_path):
    path = fixture_plan(tmp_path/'normal')
    assert start(path,MockAdapter(failure_at=1))['calls_settled'] == 16
    path = fixture_plan(tmp_path/'mismatch'); fake = MockAdapter(mismatch_at=1,unknown_at=1)
    assert start(path,fake)['calls_uncertain'] == 1
    with pytest.raises(runtime.RuntimeErrorSafe,match='contract failure'):
        runtime.execute(path,resume=True,adapter=fake)
    assert len(fake.sent) == 1


@pytest.mark.parametrize('text,kind',[('```json\n{}\n```','parse_failure'),('{"other":[]}','validation_failure')])
def test_known_bad_output_continues_all_conditions(tmp_path,text,kind):
    path = fixture_plan(tmp_path)
    result = start(path,MockAdapter(text=text))
    assert result['calls_settled'] == 16 and result['known_model_output_failures'] == 16
    assert all(r['failure_kind'] == kind for r in ledger(path)['calls'])


def test_pricing_expiry_before_approval_and_each_send(tmp_path,monkeypatch):
    path = fixture_plan(tmp_path/'before'); called = []
    original = pricing.source.now
    monkeypatch.setattr(pricing.source,'now',lambda:original()+dt.timedelta(days=2))
    with pytest.raises(pricing.source.EvidenceError,match='expired'):
        runtime.execute(path,adapter=MockAdapter(),confirmation=lambda p:called.append(True))
    assert called == [] and not (path.parent/'execution.lock').exists()
    monkeypatch.setattr(pricing.source,'now',original)
    path = fixture_plan(tmp_path/'during')
    def before(number,*args):
        if number == 1:
            monkeypatch.setattr(pricing.source,'now',lambda:original()+dt.timedelta(days=2))
    fake = MockAdapter(before=before)
    result = start(path,fake)
    assert result['calls_settled'] == 1 and result['calls_not_sent'] == 15 and len(fake.sent) == 1
    assert ledger(path)['calls'][1]['reservation_usd'] == '0.165536'


def test_orphan_response_not_sent_refuses_resume(tmp_path):
    path = fixture_plan(tmp_path,cap='0.34')
    start(path)
    row = ledger(path)['calls'][2]
    (path.parent/'enabled/responses'/(row['call_id']+'.raw.json')).write_bytes(b'{}')
    with pytest.raises(runtime.RuntimeErrorSafe,match='orphan'):
        runtime.execute(path,resume=True,adapter=MockAdapter())


def test_insufficient_pair_budget_sends_neither_condition(tmp_path):
    path = fixture_plan(tmp_path,cap='0.2'); fake = MockAdapter()
    result = start(path,fake)
    assert result['calls_settled'] == 0 and result['calls_not_sent'] == 16
    assert result['budget']['old_budget_credit_used_usd'] == '0'
    assert result['budget']['mini_known_cost_usd_separate'] == '0.01865750'
    assert runtime._decimal(result['budget']['known_cost_usd']) == 0
    assert len(fake.sent) == 0


def test_pair_reserved_together_and_next_pair_is_not_partially_started(tmp_path):
    path = fixture_plan(tmp_path,cap='0.34')
    def inspect(number,*args):
        current = ledger(path)['calls']
        assert current[0]['reservation_usd'] == current[1]['reservation_usd'] == '0.165536'
        if number == 1:
            assert current[1]['state'] == 'NOT_SENT'
    fake = MockAdapter(before=inspect); result = start(path,fake)
    assert result['calls_settled'] == 2 and result['calls_not_sent'] == 14
    assert runtime._decimal(result['budget']['known_cost_usd']) == runtime._decimal('0.000085')
    assert all(r['reservation_usd']=='0' for r in ledger(path)['calls'][2:])


def test_unknown_reservation_cannot_be_released_from_unsent_partner(tmp_path):
    path = fixture_plan(tmp_path); fake = MockAdapter(unknown_at=1)
    start(path,fake)
    state = ledger(path); state['calls'][1]['reservation_usd'] = '0'
    runtime._save(path.parent/'enabled/ledger.json',state)
    with pytest.raises(runtime.RuntimeErrorSafe,match='partial pair reservation'):
        runtime.execute(path,resume=True,adapter=fake)
    assert len(fake.sent) == 1


@pytest.mark.parametrize('key',['mini_execution_lock','mini_completion_lock'])
def test_completed_mini_locks_exclude_canary_before_approval(tmp_path,key):
    path = fixture_plan(tmp_path); plan = runtime.verify_plan(path); approved = []
    with open(plan['source_bindings'][key]['path'],'rb') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(runtime.RuntimeErrorSafe,match='already active'):
            runtime.execute(path,adapter=MockAdapter(),confirmation=lambda p:approved.append(True))
    assert approved == [] and not (path.parent/'enabled').exists()


def test_all_old_locks_remain_held_during_send_and_checkpoint(tmp_path):
    path = fixture_plan(tmp_path)
    def check():
        plan = runtime.verify_plan(path)
        for key in ('parent_execution_lock','mini_execution_lock','mini_completion_lock'):
            with open(plan['source_bindings'][key]['path'],'rb') as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fake = MockAdapter(before=lambda *args:check())
    def checkpoint(*args):
        check()
        return fixture_checkpoint(*args)
    result = runtime.execute(path,adapter=fake,confirmation=lambda p:True,checkpoint_callback=checkpoint)
    assert result['calls_settled'] == 16


def test_auth_failure_stops_with_pair_hold_without_any_post(tmp_path):
    path = fixture_plan(tmp_path)
    class NoAuth(MockAdapter):
        def send_wire(self,*args,**kwargs):
            raise wire_adapter.ProviderError('missing_api_key')
    fake = NoAuth(); result = start(path,fake)
    assert result['status'] == 'SAFE_STOP' and result['calls_uncertain'] == 1
    assert result['calls_not_sent'] == 15 and fake.sent == []
    assert result['budget']['unsettled_and_unknown_holds_usd'] == '0.331072'
    assert ledger(path)['calls'][0]['failure_kind'] == 'missing_api_key'


def test_http_error_saves_raw_then_known_usage_and_stops(tmp_path):
    path = fixture_plan(tmp_path)
    class HTTPError(MockAdapter):
        def send_wire(self,*args,**kwargs):
            result = super().send_wire(*args,**kwargs)
            raise wire_adapter.TransportError('http_failure',503,http_response_received=True,
                                              raw_response=result['raw'],raw_complete=True)
    fake = HTTPError(); result = start(path,fake); row = ledger(path)['calls'][0]
    assert result['status'] == 'SAFE_STOP' and result['calls_settled'] == 1 and len(fake.sent) == 1
    assert row['http_status'] == 503 and row['response_received'] and row['usage_known']
    assert planning.descriptor(row['raw']['path']) == row['raw']
    assert runtime._decimal(result['budget']['known_cost_usd']) == runtime._decimal('0.0000425')
    assert result['budget']['unsettled_and_unknown_holds_usd'] == '0.165536'


def test_unknown_pair_resume_does_not_spend_reserved_partner_twice(tmp_path):
    path = fixture_plan(tmp_path,cap='0.34'); fake = MockAdapter(unknown_at=1)
    start(path,fake); fake.unknown_at = None
    result = runtime.execute(path,resume=True,adapter=fake,checkpoint_callback=fixture_checkpoint)
    assert len(fake.sent) == 2 and result['calls_settled'] == 1 and result['calls_uncertain'] == 1
    assert result['calls_not_sent'] == 14
    assert result['budget']['unsettled_and_unknown_holds_usd'] == '0.165536'
    assert all(r['reservation_usd']=='0' for r in ledger(path)['calls'][2:])


def test_parent_lock_and_parent_change_during_approval(tmp_path):
    path = fixture_plan(tmp_path); plan = runtime.verify_plan(path)
    with open(plan['source_bindings']['parent_execution_lock']['path'],'rb') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(runtime.RuntimeErrorSafe,match='already active'):
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
    with pytest.raises(planning.PlanError,match='sixteen'):
        runtime.preview(path)
    path = fixture_plan(tmp_path/'c'); start(path)
    authority = path.parent/'enabled/authority.json'
    value = json.loads(authority.read_text()); value['scope_calls'] = 17; authority.write_text(json.dumps(value))
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
    assert start(path,Inspect())['calls_settled'] == 16


def _pty_call(path, answer, *, execute=False):
    pid,master = pty.fork()
    if pid == 0:
        try:
            if execute:
                result = runtime.execute(path,adapter=MockAdapter(),checkpoint_callback=fixture_checkpoint)
                assert result['calls_settled'] == 16
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
    assert len(ledger(path)['calls']) == 16


def test_non_tty_read_failure(tmp_path,monkeypatch):
    path = fixture_plan(tmp_path)
    monkeypatch.setattr('builtins.open',lambda *a,**k: (_ for _ in ()).throw(OSError()))
    with pytest.raises(runtime.RuntimeErrorSafe,match='TTY unavailable'):
        runtime.confirm_tty(json.loads(path.read_text()))
