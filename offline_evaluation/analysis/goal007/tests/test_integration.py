from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from offline_evaluation.analysis.goal007 import integration as subject
from offline_evaluation.continuation_v1 import aggregate
from offline_evaluation.server_v2.core import Grid, evaluate_pair, grid_to_payload
from offline_evaluation.server_v2.generation import object_hash, digest, GenerationError
from offline_evaluation.server_v2.data import sha256_file


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    from offline_evaluation.server_v2 import providers
    from offline_evaluation.resmoke_v2 import adapter
    def forbidden(*a,**k): raise AssertionError('research transport must never be called')
    monkeypatch.setattr(providers,'send_request',forbidden)
    if hasattr(adapter,'send_request'): monkeypatch.setattr(adapter,'send_request',forbidden)
    full=[]; gt=Grid({(0,0,0):'stone',(1,0,0):'stone'}); pred=Grid({(1,0,0):'stone'})
    measured=evaluate_pair(gt,pred,alignment_mode='strict_translation')
    gt_payload=grid_to_payload(gt); descriptor={'building_directory':str(tmp_path/'gt'),'grid_sha256':object_hash(gt_payload)}
    models={'openai':'gpt-5-mini','anthropic':'claude-haiku-4-5-20251001'}
    for building in range(60):
        scene=f'fixture_building_{building:03d}'
        cells=[(c,d) for d in ('openai','anthropic','gemini') for c in ('O','J')]+[('P',None),('E',None)]
        offset=2 if building==58 else 0
        cells=cells[offset:]+cells[:offset]
        for c,d in cells:
            for provider in models:
                index=len(full); text=f'fixture input {building} {c} {d}'
                full.append({'schedule_index':index,'scene_id':scene,'provider':provider,'model':models[provider],
                    'condition':c,'description_model':d,'input_id':f'fixture-{index}','input_text':text,
                    'input_sha256':digest(text.encode()),'protocol_id':'fixture_fixed60','prompt_sha256':digest(b'fixture'),
                    'schema_sha256':object_hash({})})
    gt_by_scene={r['scene_id']:descriptor for r in full}
    mi={'protocol_id':'fixture_fixed60','full_schedule':full,'evaluation_scene_ids':list(gt_by_scene),'gt_by_scene':gt_by_scene,
        'prompt_sha256':digest(b'fixture'),'schema_sha256':object_hash({})}
    manifest={'identity':mi,'identity_hash':object_hash(mi)}; sources={}
    original_id='attempt-original781-940'
    for label in subject.SOURCE_LABELS:
        directory=tmp_path/label; directory.mkdir()
        schedule=[deepcopy(full[n]) for n in subject.EXPECTED_SCHEDULES[label]]
        identity={'protocol_id':label,'semantic_protocol_id':'fixture_fixed60','schedule':schedule,
                  'manifest_identity_hash':manifest['identity_hash'],'prompt':'fixture','prompt_sha256':mi['prompt_sha256'],
                  'schema_sha256':mi['schema_sha256'],'gt_by_scene':gt_by_scene,'budgets':{'max_output_tokens':32768},
                  'pricing':{p:{} for p in models},'enabled_directory':str(directory),
                  'parent':{'usage':{'calls':30,'cost_usd_upper_bound':1.0}}}
        if label=='supplement3': identity['intents']=[{'source_attempt_id':original_id}]
        if label=='supplement17':
            identity['parent']['usage']['cost_usd_upper_bound']=float(Decimal('1')+sum(
                (Decimal(str(e['charged_cost_usd'])) for source in sources.values() for e in source['ledger']['entries']),Decimal(0)))
        plan={'identity':identity,'identity_hash':object_hash(identity)}; entries=[]
        for r in schedule[:subject.EXPECTED_ATTEMPTS[label]]:
            n=r['schedule_index']; attempt=f'attempt-{label}-{n}'
            unknown=(label=='original781' and n==940) or n==942
            failed=n in {10,170} or unknown
            artifacts={}
            if not unknown:
                for name,payload in [('raw.json',{'fixture':True}),('parsed.json',{'fixture':True}),('voxels.json',grid_to_payload(pred))]:
                    if failed and name!='raw.json': continue
                    path=directory/'attempts'/attempt/name; path.parent.mkdir(parents=True,exist_ok=True)
                    subject.write_new_json(path,payload)
                    artifacts[name]={'path':str(path.relative_to(directory)),'bytes':path.stat().st_size,'sha256':sha256_file(path)}
            entries.append({'attempt_id':attempt,'schedule_index':n,'provider':r['provider'],'input_id':r['input_id'],
                'status':'failed' if failed else 'success','failure_kind':'auth_failure' if label=='original781' and n==940 else
                    'http_failure' if n==942 else 'truncation' if failed else None,
                'http_status_code':401 if label=='original781' and n==940 else 520 if n==942 else None,
                'usage':None if unknown else {'input_tokens':1,'output_tokens':1},'response_id':None if unknown else 'fixture-response',
                'response_model':None if unknown else r['model'],'artifacts':artifacts,'api_response_received':True,
                'request_payload_hash':object_hash(subject.frozen_runner._payload(identity,r)),
                'charged_cost_usd':0.165536 if unknown else 0.001,'charged_input_tokens':400000 if unknown else 1,
                'charged_output_tokens':32768 if unknown else 1,'billing':'worst_case_reservation' if unknown else 'known_usage_conservative_cost',
                'retry':label=='supplement3' and n==940,'intent':'manual_auth_retry' if label=='supplement3' and n==940 else 'first_attempt',
                'source_attempt_id':original_id if label=='supplement3' and n==940 else None})
        ledger={'plan_identity_hash':plan['identity_hash'],'entries':entries,'halt_reason':None}
        sources[label]={'plan':plan,'ledger':ledger,'plan_path':str(directory/'plan.json'),'directory':directory,
                        'evaluation':None,'evaluation_path':None}
        if label in {'parent160','original781'}:
            rows=aggregate._joined_rows(plan,None,ledger,mi['protocol_id'])
            for row in rows:
                if row['status']=='success': row.update(metrics=deepcopy(measured),valid_nonempty_reconstruction=True)
            sources[label]['evaluation']={'plan_identity_hash':plan['identity_hash'],'rows':rows}
            sources[label]['evaluation_path']=str(directory/'evaluation.json')
    return manifest,sources,gt,pred


def test_exact960_961_retry_missing_and_conditions(cohort):
    manifest,sources,_,_=cohort
    before=deepcopy((manifest,sources)); result=subject.integrate(manifest,sources)
    assert (manifest,sources)==before
    assert len(result['rows'])==960 and len(result['attempts'])==961
    assert result['validation']['condition_counts']=={'O':360,'J':360,'P':120,'E':120}
    retry=result['rows'][940]; missing=result['rows'][942]
    assert retry['selection_reason']=='approved_manual_auth_retry_after_401'
    assert retry['manual_retry'] is True and len(retry['attempt_history_ids'])==2
    assert len([r for r in result['attempts'] if r['global_schedule_index']==940])==2
    assert missing['status']=='operational_unmeasurable' and missing['metrics'] is None and missing['usage'] is None
    assert missing['completed_request_outcome'] is False
    assert subject.analysis_v4.bounded_all_attempted_value(missing) is None
    assert subject.analysis_v4.bounded_all_attempted_value(result['rows'][10])==0
    assert result['rows'][10]['metrics'] is None


@pytest.mark.parametrize('tamper',['retry_source','retry_payload','retry_flag','extra_attempt','missing_usage','missing_artifact'])
def test_reject_invalid_retry_missing_and_attempt_inventory(cohort,tamper):
    manifest,sources,_,_=cohort; b=sources['supplement3']['ledger']['entries']
    if tamper=='retry_source': b[0]['source_attempt_id']='wrong'
    elif tamper=='retry_payload': b[0]['request_payload_hash']='f'*64
    elif tamper=='retry_flag': b[0]['retry']=False
    elif tamper=='extra_attempt': b.append(deepcopy(b[0]))
    elif tamper=='missing_usage': b[2]['usage']={'input_tokens':0,'output_tokens':0}
    else: b[2]['artifacts']={'raw.json':{'path':'fake','sha256':'f'*64,'bytes':1}}
    with pytest.raises(GenerationError): subject.integrate(manifest,sources)


def test_new_metrics_only_gt_direction_and_source_immutability(cohort,tmp_path):
    manifest,sources,gt,pred=cohort; result=subject.integrate(manifest,sources)
    before={str(p):sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    calls=[]
    def measure(a,b,**kwargs):
        assert a==gt and b==pred and kwargs=={'alignment_mode':'strict_translation'}
        calls.append(1); return evaluate_pair(a,b,**kwargs)
    audit=subject.evaluate_missing(result,sources,tmp_path/'independent_cache',gt_loader=lambda _:SimpleNamespace(grid=gt),evaluator=measure)
    assert audit['computed_count']==19 and len(calls)==19 and audit['reused_count']==938
    assert all(sha256_file(Path(p))==h for p,h in before.items())
    assert len(list((tmp_path/'independent_cache').glob('*.json')))==19
    assert result['rows'][940]['metrics']['absolute']['add_count']==1
    assert result['rows'][940]['metrics']['absolute']['delete_count']==0
    assert result['rows'][942]['metrics'] is None


def test_resource_limit_preserved_not_approximated(cohort,tmp_path):
    manifest,sources,gt,_=cohort; result=subject.integrate(manifest,sources)
    def limited(*a,**k): raise subject.AlignmentResourceLimit('fixture_budget',{'fixture':'budget'})
    subject.evaluate_missing(result,sources,tmp_path/'limits',gt_loader=lambda _:SimpleNamespace(grid=gt),evaluator=limited)
    assert result['rows'][940]['status']=='ALIGNMENT_RESOURCE_LIMIT'
    assert result['rows'][940]['metrics'] is None
    assert subject.analysis_v4.bounded_all_attempted_value(result['rows'][940]) is None


def test_cached_non_strict_metrics_rejected(cohort):
    manifest,sources,_,_=cohort
    sources['parent160']['evaluation']['rows'][0]['metrics']['alignment']['globally_optimal_unrestricted']=False
    with pytest.raises(GenerationError): subject.integrate(manifest,sources)


def test_cached_resource_limit_keeps_original_cause(cohort):
    manifest,sources,_,_=cohort
    row=sources['parent160']['evaluation']['rows'][0]
    row.update(status='ALIGNMENT_RESOURCE_LIMIT',metrics=None,evaluation_failure={'kind':'ALIGNMENT_RESOURCE_LIMIT','metadata':{'reason':'preserved'}})
    actual=subject.integrate(manifest,sources)['rows'][0]
    assert actual['status']=='ALIGNMENT_RESOURCE_LIMIT' and actual['generation_status']=='success'
    assert actual['evaluation_failure']['metadata']['reason']=='preserved'
    assert subject.analysis_v4.bounded_all_attempted_value(actual) is None


def test_accounting_recomputes_all_attempts_and_preserves_both_holds(cohort):
    manifest,sources,_,_=cohort
    integrated=subject.integrate(manifest,sources)
    report=subject.cost_audit(sources,integrated['attempts'])
    assert report['historical_calls']==991 and report['fixed_scope_attempts']==961
    assert report['combined_budget_usd']=='2.290072'
    assert [h['reservation_usd'] for h in report['unknown_usage_holds']]==['0.165536','0.165536']
    assert report['actual_invoice_usd'] is None


def test_overwrite_refused_and_gate_required(tmp_path):
    path=tmp_path/'once.json'; subject.write_new_json(path,{'value':1})
    with pytest.raises(FileExistsError): subject.write_new_json(path,{'value':2})
    assert json.loads(path.read_text())=={'value':1}
    with pytest.raises(GenerationError): subject.load_sources({'status':'FAILED'})
