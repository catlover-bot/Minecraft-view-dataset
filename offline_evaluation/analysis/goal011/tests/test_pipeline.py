"""Independent synthetic checks: no historical source load or new scoring."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from offline_evaluation.analysis.goal011 import pipeline as p, source_guard as g
from offline_evaluation.analysis.goal011.tradeoff import AuditFailure
from offline_evaluation.analysis.goal007.audit import AuditError, no_python_network


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def desc(path):
    return {'bytes':path.stat().st_size, 'sha256':g.sha(path)}


def receipt(root, stage, bound, products=None):
    value={'status':'PASS','stage':stage,'binding':bound,'run_root':str(root),
           'source_package_sha256':g.SOURCE_SHA,'products':products or {},'result':{'status':'PASS'}}
    write(root/f'{stage}_{p.STAGES[stage]}_RECEIPT.json',value)
    return value


@pytest.mark.parametrize('path', ['relative', '/tmp/goal011-20261017T000000Z',
    '/root/minecraft-runs/goal010-20261017T000000Z',
    '/root/minecraft-runs/goal011-20261017T000000Z/nested',
    '/root/minecraft-runs/goal011-20261017T000000Z/../goal011-20261017T000001Z'])
def test_refuses_noncanonical_or_historical_run_paths(path):
    with pytest.raises(ValueError):p.safe_root(path,0)


def test_protected_inventory_checks_size_hash_symlink_and_membership(tmp_path):
    source=tmp_path/'immutable.json';write(source,{'value':'old'})
    inventory={};g._merge(inventory,source);g.verify_inventory(inventory)
    before=deepcopy(inventory);write(source,{'value':'new'})
    with pytest.raises(ValueError,match='hash mismatch'):g.verify_inventory(inventory)
    assert inventory==before
    source.unlink()
    with pytest.raises((AuditError,ValueError,FileNotFoundError)):g.verify_inventory(inventory)
    actual=tmp_path/'other.json';write(actual,{'value':'old'});source.symlink_to(actual)
    with pytest.raises((AuditError,ValueError)):g.verify_inventory(inventory)


def test_tracked_inventory_exempts_only_new_goal011_trees(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(g,'ROOT',tmp_path)
    names=['frozen/source.json','offline_evaluation/analysis/goal011/pipeline.py',
           'reports/server_only/goal011/README.md','reports/server_only/goal010/README.md']
    for name in names:write(tmp_path/name,{'preserved':True})
    def listing(argv,**kwargs):
        assert argv==['git','ls-files','-z'] and kwargs['cwd']==tmp_path
        return SimpleNamespace(stdout=('\0'.join(names)+'\0').encode())
    monkeypatch.setattr(g.subprocess,'run',listing)
    assert g.protected_tracked_paths()==[tmp_path/names[0],tmp_path/names[3]]


def test_binding_requires_identical_plan_and_full_production_allowlist(tmp_path,monkeypatch):
    monkeypatch.setattr(g,'ROOT',tmp_path)
    for rel in g.PRODUCTION:write(tmp_path/rel,{'fixture':True})
    def verify(commit, files):
        return {name: ('b'*64 if commit=='spec' and name==g.PLAN else 'a'*64) for name in files}
    monkeypatch.setattr(g.inherited,'verify_commit',verify)
    with pytest.raises(ValueError,match='definitions differ'):g.binding('code','spec')
    bound=g.binding('code','code');assert bound['analysis_plan_sha256']==bound['plan_sha256']
    write(tmp_path/'offline_evaluation/analysis/goal011/unbound.py',{})
    with pytest.raises(ValueError,match='allowlist'):g.binding('code','code')


@pytest.mark.parametrize('change',['stage','binding','root','source','status','product'])
def test_prior_receipts_reject_scope_binding_or_product_changes(tmp_path,change):
    bound={'plan':'frozen'};product=tmp_path/'source.json';write(product,{'n':1})
    value=receipt(tmp_path,0,bound,{product.name:desc(product)})
    assert len(p.previous(tmp_path,1,bound))==1
    if change=='stage':value['stage']=1
    elif change=='binding':value['binding']={'plan':'other'}
    elif change=='root':value['run_root']=str(tmp_path/'other')
    elif change=='source':value['source_package_sha256']='f'*64
    elif change=='status':value['status']='STOP'
    else:write(product,{'n':2})
    write(tmp_path/'0_source_RECEIPT.json',value)
    with pytest.raises(ValueError):p.previous(tmp_path,1,bound)


@pytest.fixture
def orchestration(tmp_path,monkeypatch):
    target=tmp_path/'newrun';package={'immutable':'fixture'}
    bound={'code_commit':'c'*40,'specification_commit':'d'*40,'plan_sha256':'a'*64,
           'analysis_plan_sha256':'a'*64,'question_register_sha256':'b'*64,'hashes':{}}
    protected=tmp_path/'protected/source.json';write(protected,{'immutable':1})
    inventory={str(protected):desc(protected)}
    state={'active':False,'budget':False,'events':[], 'exit_fail':False,'load_fail':False,
           'late_change':False,'critical':False,'stage':0}
    monkeypatch.setattr(p,'safe_root',lambda value,stage:target)
    monkeypatch.setattr(g,'binding',lambda *args:deepcopy(bound))
    monkeypatch.setattr(g,'SOURCE_SHA',p.old.digest(package))
    monkeypatch.setattr(p.resource,'setrlimit',lambda *args:None)
    @contextmanager
    def budget(cpu,wall):
        assert (cpu,wall)==(1800,3600);state['budget']=True
        try:yield
        finally:state['budget']=False
    monkeypatch.setattr(p,'sample_budget',budget)
    def active(event):
        assert state['active'] and state['budget'];state['events'].append(event)
        with pytest.raises(AuditError,match='python_network_disabled'):
            socket.socket.connect(None,('fixture.invalid',1))
    @contextmanager
    def locked():
        with no_python_network():
            state['active']=True;active('lock_enter')
            try:
                yield ['read-only-fixture-lock'];active('exit_process_audit')
                assert not (target/f"{state['stage']}_{p.STAGES[state['stage']]}_RECEIPT.json").exists()
                if state['exit_fail']:raise ValueError('fixture exit process audit failed')
            finally:state['active']=False
    monkeypatch.setattr(g.inherited,'locked_sources',locked)
    def load():
        active('load_sources')
        if state['load_fail']:raise ValueError('fixture initial protected hash mismatch')
        return deepcopy(package),tmp_path/'old010',tmp_path/'oldv2',deepcopy(inventory),{
            'status':'PASS','source_package_sha256':g.SOURCE_SHA}
    monkeypatch.setattr(g,'load_sources',load)
    def after():active('recheck_original');return deepcopy(package)
    monkeypatch.setattr(g.inherited,'load_verified_sources',after)
    original=p.stage_work
    def stage_work(*args):
        active('stage_work')
        if state['critical']:raise AuditFailure('FIXTURE_SAVED_VALUE_CHANGED',{'slot':7})
        result=original(*args)
        if state['late_change']:write(protected,{'immutable':2})
        return result
    monkeypatch.setattr(p,'stage_work',stage_work)
    return target,state,bound,inventory


def test_full_source_orchestration_keeps_network_locks_budget_until_exit(orchestration):
    target,state,_,_=orchestration
    result=p.run(0,str(target),'code','spec')
    assert result['status']=='PASS' and result['source_lock_count']==1
    assert result['python_network_connections_disabled'] and not result['OS_level_isolation_guaranteed']
    assert state['events']==['lock_enter','load_sources','stage_work','recheck_original','exit_process_audit']
    assert not state['active'] and not state['budget']
    assert (target/'0_source_RECEIPT.json').is_file()
    assert all(result[k]==0 for k in p.ZERO)


def test_initial_failure_has_no_started_file_or_output(orchestration):
    target,state,_,_=orchestration;state['load_fail']=True
    with pytest.raises(ValueError,match='initial protected'):p.run(0,str(target),'code','spec')
    assert not target.exists() and not state['active']


@pytest.mark.parametrize('failure',['exit_fail','late_change','critical'])
def test_stage_failure_preserves_evidence_and_never_writes_pass(orchestration,failure):
    target,state,_,_=orchestration;state[failure]=True
    with pytest.raises(ValueError):p.run(0,str(target),'code','spec')
    assert not (target/'0_source_RECEIPT.json').exists()
    assert (target/'0_STARTED.json').is_file()
    stop=g.read_json(target/'0_STOP.json');assert stop['status']=='STOP'
    assert stop['historical_mutations'] is None
    assert stop['historical_integrity_status']=='NOT_VERIFIED_STAGE_FAILED'
    assert stop['historical_write_operations_by_pipeline']==0
    if failure=='critical':
        assert stop['critical_evidence']['code']=='FIXTURE_SAVED_VALUE_CHANGED'
        assert stop['critical_evidence']['evidence']=={'slot':7}
        assert stop['critical_evidence']['historical_changes_permitted'] is False
    assert not state['active'] and not state['budget']


def test_partial_stage_never_overwritten(orchestration):
    target,state,_,_=orchestration;kept=target/'0_source/retained.json';write(kept,{'keep':True});before=desc(kept)
    with pytest.raises(FileExistsError):p.run(0,str(target),'code','spec')
    assert desc(kept)==before and state['events']==[]


def test_nonpassing_stage_result_cannot_receive_pass_receipt(orchestration,monkeypatch):
    target,state,_,_=orchestration
    monkeypatch.setattr(p,'stage_work',lambda *args:{'status':'STOP','reason':'fixture'})
    with pytest.raises(ValueError):p.run(0,str(target),'code','spec')
    assert not (target/'0_source_RECEIPT.json').exists()
    assert g.read_json(target/'0_STOP.json')['status']=='STOP'


def test_code_binding_changed_midstage_stops_without_pass(orchestration,monkeypatch):
    target,state,bound,_=orchestration;calls=[]
    def binding(*args):
        calls.append(True)
        return bound if len(calls)==1 else {**bound,'plan_sha256':'f'*64}
    monkeypatch.setattr(g,'binding',binding)
    with pytest.raises(ValueError,match='code changed'):p.run(0,str(target),'code','spec')
    assert not (target/'0_source_RECEIPT.json').exists() and (target/'0_STOP.json').exists()


def test_next_stage_rejects_changed_initial_inventory_before_started(orchestration):
    target,state,bound,inventory=orchestration
    receipt(target,0,bound);write(target/'0_source/PROTECTED_INVENTORY_BEFORE.json',{})
    state['stage']=1
    with pytest.raises(ValueError,match='inventory changed'):p.run(1,str(target),'code','spec')
    assert not (target/'1_STARTED.json').exists() and not (target/'1_tradeoff').exists()


def test_rest_to_prospective_stage_contract_preserves_typed_sources(tmp_path,monkeypatch):
    from offline_evaluation.analysis.goal011 import rest, prospective
    package={'fixed':'package'};gate={'fixture':'gate'};proof={};inventory={}
    rest_result={'feasibility':{'independent_rest_buildings':30},'sources':[
        {'scene_id':'fixture','selected_for_prospective_design':True,'size_rank_stratum':0,
         'input_characters':7,'input_utf8_bytes':11,'actual_description_request_view_count':None,
         'prior_exposure_evidence':[{'kind':'fixture'}]}], 'inventory':{}}
    monkeypatch.setattr(rest,'audit_rest30',lambda *args,**kwargs:deepcopy(rest_result))
    monkeypatch.setattr(rest,'verify_inventory',lambda value:None)
    a=tmp_path/'3_rest';a.mkdir()
    p.stage_work(3,tmp_path,a,package,tmp_path/'old',tmp_path/'v2',gate,proof,inventory)
    tradeoff={'claim_audit':{'status':'PASS'}};write(tmp_path/'1_tradeoff/RESULT.json',tradeoff)
    protocol={'status':'UNAPPROVED_NOT_RUN_NO_LAUNCHER','execution_command':None}
    def design(pkg,trade,rest_value,gate):
        assert pkg==package and trade==tradeoff
        assert rest_value['feasibility']==rest_result['feasibility']
        assert rest_value['sources']==rest_result['sources']
        return {'protocol':protocol,'report_markdown':'Unapproved; no execution.'}
    monkeypatch.setattr(prospective,'design_protocol',design)
    b=tmp_path/'4_prospective';b.mkdir()
    result=p.stage_work(4,tmp_path,b,package,tmp_path/'old',tmp_path/'v2',gate,proof,inventory)
    assert result['status']=='PASS'
    assert g.read_json(b/'PROSPECTIVE_DESIGN.json')==protocol
    assert (b/'PROSPECTIVE_PROTOCOL_UNAPPROVED.md').read_text()=='Unapproved; no execution.'


@pytest.fixture
def publish_fixture(tmp_path,monkeypatch):
    target=tmp_path/'canonical';report=tmp_path/'public';report.mkdir()
    bound={'plan':'frozen'}
    names={'1_tradeoff':('CLAIM_AUDIT_FINDINGS.json','TRADEOFF_EXPLANATION_JA.md','global_saved_effects.csv',
                         'quintile_effects.csv','valid_mask_audit.csv','precision_design_inputs.csv'),
           '2_material':('E_CASES_101.csv','E_GLOBAL_CONFUSION.csv','E_MAPPING_UPPER_BOUNDS.csv',
                         'E_FAILURE_MECHANISM_AUDIT.json','E_SYMBOL_BINDING_FINDINGS_JA.md'),
           '3_rest':('REST30_FEASIBILITY.json','REST30_SOURCES.csv'),
           '4_prospective':('PROSPECTIVE_DESIGN.json','PROSPECTIVE_PROTOCOL_UNAPPROVED.md')}
    for folder,files in names.items():
        for name in files:write(target/folder/name,{'safe_metadata':name})
    for i,stage in enumerate(p.STAGES):write(target/f'{i}_{stage}/PROTECTED_AFTER.json',{'status':'PASS'})
    raw=target/'2_material/raw_response.json';write(raw,{'raw':'must remain private'})
    for i in range(5):receipt(target,i,bound,{str(raw.relative_to(target)):desc(raw)})
    monkeypatch.setattr(p,'safe_root',lambda *args:target)
    monkeypatch.setattr(g,'binding',lambda *args:bound)
    monkeypatch.setattr(g,'REPORT',report)
    return target,report


def test_material_stage_exports_global_counts_and_sanitized_diagnostics(tmp_path,monkeypatch):
    from offline_evaluation.analysis.goal011 import material_batch
    old_root=tmp_path/'old'; control=old_root/'phase3_exact_oracle/E_reference_controls.json'
    write(control,{'fixture':True})
    out=tmp_path/'2_material';out.mkdir()
    result={'cases':[{'case':i} for i in range(101)],'confusion':[{'unsafe_case_rows':1}],
            'global_confusion':[{'voxel_count':17,'weighting':'POOLED_VOXELS_NOT_EQUAL_BUILDING'}],
            'mapping':[],'audit':{'status':'PASS','all_case_confusions_retained':True},
            'source_inventory':{},'report_markdown':'fixture'}
    monkeypatch.setattr(material_batch,'run_material',lambda *args,**kwargs:result)
    got=p.stage_work(2,tmp_path,out,{},old_root,tmp_path/'v2',{}, {},{})
    assert got['cases']==101
    assert g.read_json(out/'E_FAILURE_MECHANISM_AUDIT.json')==result['audit']
    assert 'POOLED_VOXELS_NOT_EQUAL_BUILDING' in (out/'E_GLOBAL_CONFUSION.csv').read_text()
    assert 'unsafe_case_rows' not in (out/'E_GLOBAL_CONFUSION.csv').read_text()


def test_publish_exact_metadata_allowlist_and_no_raw_copy(publish_fixture):
    target,report=publish_fixture;before=desc(target/'2_material/raw_response.json')
    p.publish(str(target),'code','spec')
    receipt=g.read_json(report/'PUBLIC_COPY_RECEIPT.json')
    assert len(receipt['files'])==25 and receipt['raw_responses_copied']==0 and receipt['GT_arrays_copied']==0
    assert not any('raw_response' in x for x in receipt['files'])
    for rel,expected in receipt['files'].items():
        assert desc(report/rel)=={k:expected[k] for k in ('bytes','sha256')}
    assert desc(target/'2_material/raw_response.json')==before


@pytest.mark.parametrize('mode',['existing','oversize','symlink'])
def test_publish_preflight_rejection_creates_no_partial_copies(publish_fixture,mode):
    target,report=publish_fixture;source=target/'3_rest/REST30_SOURCES.csv'
    if mode=='existing':write(report/'REST30_SOURCES.csv',{'retained':True})
    elif mode=='oversize':source.write_bytes(b'x'*(10*1024**2+1))
    else:
        source.unlink();source.symlink_to(target/'3_rest/REST30_FEASIBILITY.json')
    before={str(f.relative_to(report)):desc(f) for f in report.rglob('*') if f.is_file()}
    with pytest.raises((ValueError,AuditError)):p.publish(str(target),'code','spec')
    assert {str(f.relative_to(report)):desc(f) for f in report.rglob('*') if f.is_file()}==before


def test_publish_rejects_drift_in_verified_prior_product(publish_fixture):
    target,report=publish_fixture;write(target/'2_material/raw_response.json',{'raw':'changed'})
    with pytest.raises(ValueError,match='hash mismatch'):p.publish(str(target),'code','spec')
    assert list(report.iterdir())==[]
