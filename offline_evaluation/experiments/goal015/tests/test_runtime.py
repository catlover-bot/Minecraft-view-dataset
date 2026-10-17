"""Fixture-only lifecycle and real controlling-PTY approval checks."""
import copy
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest
from offline_evaluation.experiments.goal015 import runtime as r


def fixture_plan(tmp_path):
    root=tmp_path/'fixture-run';root.mkdir();spec=root/'spec.md';spec.write_text('fixture specification')
    payload=root/'payload.json';payload.write_text('{"mock":true}')
    code=root/'code.py';code.write_text('# fixture implementation')
    contract={'models':{'openai':'fixture-openai','anthropic':'fixture-anthropic'},'rates_per_million':{'openai':{'input':'0.25','output':'2'},'anthropic':{'input':'1','output':'5'}},'freshness_seconds':86400}
    evidence=root/'pricing.json';evidence.write_text(json.dumps({'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'pricing_contract':contract}))
    rows=[]
    for scene in range(10):
      for b in ('openai','anthropic'):
       for repeat in (1,2):
        for condition in (('SUMMARY_ONLY','FULL_RECORD') if repeat==1 else ('FULL_RECORD','SUMMARY_ONLY')):
         rows.append({'call_id':f'c{len(rows):02d}','scene_id':f's{scene}','builder':b,'condition':condition,'repeat_index':repeat,
           'paired_block_id':f's{scene}-{b}-{repeat}','payload':r.descriptor(payload),'reserve_usd':'0.01','input_reservation_tokens':100})
    plan={'schema':'goal015_disabled_plan_v1','canonical_root':str(root),'api_allowed':False,'budget_approved':False,
      'fixture_only':True,'timeout_seconds':600,'proposed_budget_usd':'5.00','planned_calls':80,'planned_buildings':10,'concurrency':1,'max_retries':0,
      'pricing_evidence':r.descriptor(evidence),'pricing_contract':contract,'implementation_hashes':{str(code):r.descriptor(code)['sha256']},
      'specification':r.descriptor(spec),'schedule':rows};save_plan(root/'plan.json',plan);return root/'plan.json'


def save_plan(path,plan):plan['plan_id']=r.identity(plan);path.write_text(json.dumps(plan));return plan


class MockAdapter:
    def __init__(self,fail_at=None,unknown_at=None,interrupt_at=None):self.calls=0;self.fail_at=fail_at;self.unknown_at=unknown_at;self.interrupt_at=interrupt_at
    def validate_payload(self,b,p):assert p=={'mock':True}
    def send_request(self,b,p,**kwargs):
        assert kwargs=={'api_allowed':True,'permission':True,'timeout':600};self.calls+=1
        if self.calls==self.interrupt_at:raise KeyboardInterrupt()
        if self.calls==self.fail_at:raise OSError('SECRET MUST NOT BE LOGGED')
        return json.dumps({'usage_known':self.calls!=self.unknown_at}).encode()
    def parse_response(self,b,raw):
        return SimpleNamespace(complete=True,model='fixture',response_id='fixture',failure_kind=None,text='{"operations":[]}',usage=json.loads(raw))
    def classify_usage(self,b,usage,pricing):
        return {'usage_known':usage['usage_known'],'cost_usd':.001 if usage['usage_known'] else None}


def execute(path,adapter=None,**kwargs):
    return r.execute(path,adapter=adapter or MockAdapter(),confirmation=lambda plan:True,
                     checkpoint_callback=lambda *x:{'status':'LOCAL_VERIFIED','fixture_only':True},**kwargs)


def test_preview_status_no_enabled_or_keys(tmp_path,monkeypatch):
    p=fixture_plan(tmp_path);before={str(x) for x in tmp_path.rglob('*')}
    monkeypatch.setattr(os,'getenv',lambda *x: (_ for _ in ()).throw(AssertionError('key lookup')))
    assert r.preview(p)['budget_approved'] is False and r.status(p)['research_api_requests_sent']==0
    assert {str(x) for x in tmp_path.rglob('*')}==before


def test_mock80_and_duplicate_execution(tmp_path):
    p=fixture_plan(tmp_path);a=MockAdapter();out=execute(p,a)
    assert out['status']=='COMPLETE' and out['calls_settled']==a.calls==80
    ledger=r.read_json(p.parent/'enabled/ledger.json');assert len(ledger['calls'])==80
    assert all(row['raw'] and row['response_text_path'] and row['state']=='SETTLED' for row in ledger['calls'])
    with pytest.raises(r.RuntimeErrorSafe,match='existing execution'):execute(p,a)
    execute(p,a,resume=True);assert a.calls==80


def test_no_approval_no_state(tmp_path):
    p=fixture_plan(tmp_path)
    with pytest.raises(r.RuntimeErrorSafe):r.execute(p,adapter=MockAdapter(),confirmation=lambda p:False)
    assert not (p.parent/'enabled').exists() and not (p.parent/'execution.lock').exists()


def test_unknown_retains_hold_and_resume_never_resends(tmp_path):
    p=fixture_plan(tmp_path);a=MockAdapter(unknown_at=1);first=execute(p,a)
    assert first['calls_uncertain']==1 and a.calls==1
    second=execute(p,a,resume=True)
    assert a.calls==80 and second['calls_settled']==79 and second['calls_uncertain']==1
    assert second['budget']['unsettled_and_unknown_holds_usd']=='0.01'


@pytest.mark.parametrize('kind',['transport','interrupt'])
def test_uncertain_send_never_retried(tmp_path,kind):
    p=fixture_plan(tmp_path);a=MockAdapter(fail_at=1 if kind=='transport' else None,interrupt_at=1 if kind=='interrupt' else None)
    assert execute(p,a)['status']=='SAFE_STOP'
    execute(p,a,resume=True);assert a.calls==80
    assert 'SECRET' not in (p.parent/'enabled/ledger.json').read_text()


def test_whole_paired_block_budget(tmp_path):
    p=fixture_plan(tmp_path);plan=r.read_json(p)
    for row in plan['schedule'][:2]:row['reserve_usd']='3'
    save_plan(p,plan);a=MockAdapter();out=execute(p,a)
    assert out['status']=='SAFE_STOP' and a.calls==0 and out['calls_not_sent']==80


def test_stale_and_changed_pricing_refuse_before_approval(tmp_path):
    p=fixture_plan(tmp_path);plan=r.read_json(p);e=Path(plan['pricing_evidence']['path']);d=r.read_json(e)
    d['retrieved_at_utc']=(datetime.now(timezone.utc)-timedelta(hours=24,seconds=1)).isoformat();e.write_text(json.dumps(d))
    plan['pricing_evidence']=r.descriptor(e);save_plan(p,plan)
    with pytest.raises(r.RuntimeErrorSafe,match='24 hours'):execute(p)
    assert not (p.parent/'enabled').exists()


def test_copy_and_implementation_change_rejected(tmp_path):
    p=fixture_plan(tmp_path);copy_path=p.parent/'copy.json';copy_path.write_bytes(p.read_bytes())
    with pytest.raises(r.RuntimeErrorSafe,match='copied'):r.verify_plan(copy_path)
    (p.parent/'code.py').write_text('changed')
    with pytest.raises(r.RuntimeErrorSafe,match='implementation'):r.verify_plan(p)


def test_production_disallows_injected_confirmation(tmp_path,monkeypatch):
    p=fixture_plan(tmp_path);plan=r.read_json(p);plan['fixture_only']=False;save_plan(p,plan)
    monkeypatch.setattr(r,'verify_plan',lambda path:plan)
    with pytest.raises(r.RuntimeErrorSafe,match='mock hooks'):execute(p)


def test_raw_durable_before_accounting(tmp_path):
    p=fixture_plan(tmp_path)
    class Checking(MockAdapter):
        def classify_usage(self,*args):
            ledger=r.read_json(p.parent/'enabled/ledger.json');row=next(row for row in ledger['calls'] if row['state']=='SENT_PENDING')
            assert row['raw'] and r.descriptor(row['raw']['path'])==row['raw']
            return super().classify_usage(*args)
    assert execute(p,Checking())['calls_settled']==80


def test_canonical_lock_rejects_second_process(tmp_path):
    p=fixture_plan(tmp_path)
    with r.canonical_lock(p.parent):
        code='from offline_evaluation.experiments.goal015.runtime import canonical_lock\nwith canonical_lock('+repr(str(p.parent))+'): pass'
        out=subprocess.run([sys.executable,'-B','-c',code],capture_output=True)
        assert out.returncode!=0 and b'already active' in out.stderr


def _pty_case(answer,fail_read=False):
    pid,master=pty.fork()
    if pid==0:
        if fail_read:
            import builtins
            original=builtins.open
            class Broken:
                def __init__(self,f):self.f=f
                def __enter__(self):return self
                def __exit__(self,*args):self.f.close()
                def fileno(self):return self.f.fileno()
                def readline(self,*args):raise OSError('fixture read failure')
            builtins.open=lambda path,mode,*a,**kw: Broken(original(path,mode,*a,**kw)) if path=='/dev/tty' and mode=='r' else original(path,mode,*a,**kw)
        try:r.confirm_tty({'plan_id':'f'*64,'proposed_budget_usd':'5.00'});os._exit(0)
        except BaseException:os._exit(7)
    received=b'';deadline=time.monotonic()+5
    while b'> ' not in received and time.monotonic()<deadline:
        ready,_,_=select.select([master],[],[],.1)
        if ready:
            try:received+=os.read(master,8192)
            except OSError:break
    if not fail_read:
        try:os.write(master,answer)
        except OSError:pass
    while time.monotonic()<deadline:
        done,status=os.waitpid(pid,os.WNOHANG)
        if done:os.close(master);return os.waitstatus_to_exitcode(status),received
        time.sleep(.01)
    os.kill(pid,9);os.waitpid(pid,0);os.close(master);raise AssertionError('PTY timed out')


@pytest.mark.parametrize('answer,success',[(('APPROVE GOAL015 '+'f'*64+' USD 5.00 80 CALLS RETRY 0\n').encode(),True),(b'\n',False),(b'no\n',False),(b'\x04',False)])
def test_real_pty_exact_enter_reject_eof(answer,success):
    code,output=_pty_case(answer);assert (code==0)==success and b'Type exactly:' in output


def test_real_pty_read_failure():assert _pty_case(b'',True)[0]!=0


def test_real_pty_approval_then_duplicate_start_refused_without_new_prompt(tmp_path):
    p=fixture_plan(tmp_path);plan=r.read_json(p);pid,master=pty.fork()
    if pid==0:
        try:
            a=MockAdapter()
            kwargs={'adapter':a,'checkpoint_callback':lambda *args:{'status':'LOCAL_VERIFIED','fixture_only':True}}
            assert r.execute(p,**kwargs)['calls_settled']==80
            try:r.execute(p,**kwargs)
            except r.RuntimeErrorSafe as exc:assert 'existing execution' in str(exc)
            else:raise AssertionError('duplicate start accepted')
            assert a.calls==80
            r._save(tmp_path/'REAL_PTY_DUPLICATE_RECEIPT.json',{'fixture_only':True,
                'mock_calls':a.calls,'real_api_requests':0,'real_tty_approval':True,
                'duplicate_start_refused_before_second_prompt':True},exclusive=True)
            os._exit(0)
        except BaseException:os._exit(7)
    received=b'';deadline=time.monotonic()+15;answered=False
    while time.monotonic()<deadline:
        ready,_,_=select.select([master],[],[],.05)
        if ready:
            try:received+=os.read(master,8192)
            except OSError:pass
        if not answered and b'> ' in received:
            os.write(master,(r.approval_text(plan)+'\n').encode());answered=True
        done,status=os.waitpid(pid,os.WNOHANG)
        if done:
            os.close(master)
            assert os.waitstatus_to_exitcode(status)==0
            assert received.count(b'Type exactly:')==1
            assert r.read_json(tmp_path/'REAL_PTY_DUPLICATE_RECEIPT.json')['mock_calls']==80
            return
    os.kill(pid,9);os.waitpid(pid,0);os.close(master);raise AssertionError('PTY duplicate test timed out')


def test_no_controlling_tty_fails_closed():
    code="from offline_evaluation.experiments.goal015.runtime import confirm_tty;confirm_tty({'plan_id':'f'*64,'proposed_budget_usd':'5.00'})"
    out=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,start_new_session=True)
    assert out.returncode!=0 and b'TTY' in out.stderr


@pytest.mark.parametrize('kind',['response_model_mismatch','unexpected_response_feature','incomplete_response'])
def test_contract_failure_blocks_automatic_resume(tmp_path,kind):
    p=fixture_plan(tmp_path)
    class Contract(MockAdapter):
        def parse_response(self,*args):
            response=super().parse_response(*args);response.complete=False;response.failure_kind=kind;return response
    a=Contract();out=execute(p,a)
    assert out['status']=='SAFE_STOP' and a.calls==1
    with pytest.raises(r.RuntimeErrorSafe,match='new explicit approval'):execute(p,a,resume=True)
    assert a.calls==1


def test_ledger_settled_missing_cost_refused(tmp_path):
    p=fixture_plan(tmp_path);a=MockAdapter();execute(p,a)
    path=p.parent/'enabled/ledger.json';ledger=r.read_json(path);ledger['calls'][0]['known_cost_usd']=None;path.write_text(json.dumps(ledger))
    with pytest.raises(r.RuntimeErrorSafe,match='known cost'):execute(p,a,resume=True)
    assert a.calls==80


def test_http_error_raw_usage_classified_without_model_failure(tmp_path):
    p=fixture_plan(tmp_path)
    class HTTPError(Exception):
        failure_kind='auth_failure';http_status=401;http_response_received=True;raw_response=b'{"usage_known":true}';raw_complete=True
    class Failure(MockAdapter):
        def send_request(self,*args,**kwargs):self.calls+=1;raise HTTPError()
    a=Failure();out=execute(p,a);row=r.read_json(p.parent/'enabled/ledger.json')['calls'][0]
    assert out['status']=='SAFE_STOP' and a.calls==1 and row['response_received'] is True
    assert row['http_status']==401 and row['raw'] and row['usage_known'] is True and row['state']=='SETTLED'
    assert row['model_output_failure'] is False and row['completed_response'] is False


def test_partial_http_body_never_releases_unknown_hold(tmp_path):
    p=fixture_plan(tmp_path)
    class HTTPError(Exception):
        failure_kind='http_failure';http_status=520;http_response_received=True;raw_response=b'{"usage_known":true}';raw_complete=False
    class Failure(MockAdapter):
        def send_request(self,*args,**kwargs):self.calls+=1;raise HTTPError()
    out=execute(p,Failure());row=r.read_json(p.parent/'enabled/ledger.json')['calls'][0]
    assert row['raw'] and not row['usage_known'] and row['state']=='UNKNOWN_USAGE'
    assert out['budget']['unsettled_and_unknown_holds_usd']=='0.02'


def test_stale_during_pair_stops_before_second_request(tmp_path,monkeypatch):
    p=fixture_plan(tmp_path);original=r.pricing_fresh;a=MockAdapter();count=[0]
    def timed(plan,*args,**kwargs):
        count[0]+=1
        if a.calls==1:raise r.RuntimeErrorSafe('fixture stale24 hours')
        return original(plan,*args,**kwargs)
    monkeypatch.setattr(r,'pricing_fresh',timed)
    out=execute(p,a);assert out['status']=='SAFE_STOP' and a.calls==1 and out['calls_not_sent']==79


def test_root_and_enabled_directories_fsynced_before_send(tmp_path,monkeypatch):
    p=fixture_plan(tmp_path);observed=[];original=r._fsync_directory
    def synced(path):observed.append(Path(path));return original(path)
    monkeypatch.setattr(r,'_fsync_directory',synced)
    class Checked(MockAdapter):
        def send_request(self,*args,**kwargs):
            assert p.parent in observed and p.parent/'enabled' in observed
            assert (p.parent/'enabled/authority.json').is_file()
            return super().send_request(*args,**kwargs)
    assert execute(p,Checked())['calls_settled']==80


def test_model_shutdown_stops_even_fresh_evidence(tmp_path):
    p=fixture_plan(tmp_path);plan=r.read_json(p);plan['pricing_contract']['shutdown_at_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    e=Path(plan['pricing_evidence']['path']);proof=r.read_json(e);proof['pricing_contract']=plan['pricing_contract'];e.write_text(json.dumps(proof));plan['pricing_evidence']=r.descriptor(e);save_plan(p,plan)
    with pytest.raises(r.RuntimeErrorSafe,match='shutdown'):execute(p)
    assert not (p.parent/'enabled').exists()


def test_prepared_registry_refuses_rehashed_copy(tmp_path,monkeypatch):
    p=fixture_plan(tmp_path);plan=r.read_json(p);registry=tmp_path/'binding.json'
    registry.write_text(json.dumps({'plan_id':plan['plan_id'],'canonical_root':plan['canonical_root'],'plan':r.descriptor(p)}))
    monkeypatch.setattr(r,'EXECUTION_BINDING',registry)
    r._verify_registry(plan,p)
    plan['fixture_only']=False;save_plan(p,plan)
    with pytest.raises(r.RuntimeErrorSafe,match='registry'):r._verify_registry(plan,p)


def test_contract_failure_unknown_usage_and_missing_global_flag_cannot_resume(tmp_path):
    p=fixture_plan(tmp_path)
    class Contract(MockAdapter):
        def parse_response(self,*args):
            response=super().parse_response(*args);response.complete=False;response.failure_kind='response_model_mismatch';return response
    a=Contract(unknown_at=1);out=execute(p,a)
    assert out['status']=='SAFE_STOP' and out['calls_uncertain']==1
    path=p.parent/'enabled/ledger.json';ledger=r.read_json(path)
    assert ledger['requires_new_explicit_approval'] is True
    ledger.pop('requires_new_explicit_approval');path.write_text(json.dumps(ledger))
    with pytest.raises(r.RuntimeErrorSafe,match='new explicit approval'):execute(p,a,resume=True)
    assert a.calls==1


def test_production_pricing_requires_reviewed_current_document_bytes(tmp_path):
    p=fixture_plan(tmp_path);plan=r.read_json(p);plan['fixture_only']=False
    evidence=Path(plan['pricing_evidence']['path']);proof=r.read_json(evidence)
    with pytest.raises(r.RuntimeErrorSafe,match='reviewed official'):r.pricing_fresh(plan)
    doc=tmp_path/'official.html';doc.write_text('fixture official page')
    proof.update(review_status='REVIEWED_OFFICIAL_DOCUMENTS',documents=[dict(r.descriptor(doc),
        url='https://developers.openai.com/api/docs/pricing',http_status=200,retrieved_at_utc=proof['retrieved_at_utc'])])
    evidence.write_text(json.dumps(proof));plan['pricing_evidence']=r.descriptor(evidence)
    assert r.pricing_fresh(plan)['review_status']=='REVIEWED_OFFICIAL_DOCUMENTS'
    doc.write_text('changed')
    with pytest.raises(r.RuntimeErrorSafe,match='document changed'):r.pricing_fresh(plan)


@pytest.mark.parametrize('change',['missing_source','old_doc','future_doc','newer_overall','http_failure'])
def test_refresh_requires_complete_current_official_source_set(tmp_path,change):
    p=fixture_plan(tmp_path);plan=r.read_json(p);plan['fixture_only']=False
    evidence=Path(plan['pricing_evidence']['path']);proof=r.read_json(evidence)
    doc=tmp_path/'official.html';doc.write_text('fixture official page')
    stamp=(datetime.now(timezone.utc)-timedelta(minutes=5)).isoformat()
    proof.update(retrieved_at_utc=stamp,review_status='REVIEWED_OFFICIAL_DOCUMENTS',documents=[
        dict(r.descriptor(doc),url=url,http_status=200,retrieved_at_utc=stamp) for url in
        ('https://developers.openai.com/api/docs/pricing','https://platform.claude.com/docs/en/about-claude/models')])
    evidence.write_text(json.dumps(proof));plan['pricing_evidence']=r.descriptor(evidence)
    r.pricing_fresh(plan)
    refreshed=copy.deepcopy(proof)
    if change=='missing_source':refreshed['documents'].pop()
    if change=='old_doc':refreshed['documents'][0]['retrieved_at_utc']=(datetime.now(timezone.utc)-timedelta(hours=25)).isoformat()
    if change=='future_doc':refreshed['documents'][0]['retrieved_at_utc']=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    if change=='newer_overall':refreshed['retrieved_at_utc']=datetime.now(timezone.utc).isoformat()
    if change=='http_failure':refreshed['documents'][0]['http_status']=503
    target=tmp_path/'refresh.json';target.write_text(json.dumps(refreshed))
    with pytest.raises(r.RuntimeErrorSafe):r.pricing_fresh(plan,r.descriptor(target))


def test_mock80_missing_resume_saved_evaluation_and_aggregation(tmp_path):
    from offline_evaluation.experiments.goal015 import evaluate
    p=fixture_plan(tmp_path);a=MockAdapter(unknown_at=1)
    first=r.execute(p,adapter=a,confirmation=lambda plan:True)
    assert first['status']=='SAFE_STOP' and first['calls_uncertain']==1 and a.calls==1
    final=r.execute(p,resume=True,adapter=a)
    assert final['status']=='PARTIAL_WITH_UNCERTAIN_CALLS' and final['calls_settled']==79 and a.calls==80
    plan=r.read_json(p);ledger=r.read_json(p.parent/'enabled/ledger.json');before=p.read_bytes()
    summary=evaluate.evaluate_saved(plan['schedule'],ledger['calls'],p.parent/'evaluation',
        lambda scene:evaluate.core.Grid({(0,0,0):'stone'}),fixture=True)
    assert summary['planned_calls']==80 and summary['planned_independent_buildings']==10
    assert summary['evaluated_protocol_valid_rows']==79 and summary['actual_metric_rows']==79
    assert len(summary['primary_paired_estimates'])==8 and p.read_bytes()==before and a.calls==80
    assert summary['research_api_requests_sent_by_evaluation']==0
    r._save(tmp_path/'MOCK80_MISSING_RESUME_EVALUATION_RECEIPT.json',{
        'status':'PASS','fixture_only':True,'mock_transport_calls':80,'real_research_api_requests':0,
        'first_safe_stop_after_mock_call':1,'unknown_usage_holds':1,'settled_calls':79,
        'planned_calls':80,'planned_independent_buildings':10,'protocol_valid_and_measured_rows':79,
        'paired_primary_estimands':8,'generation_retries':0,'source_plan_unchanged':True,
        'evaluation_api_requests':0,'unknown_usage_not_imputed_zero':True},exclusive=True)
