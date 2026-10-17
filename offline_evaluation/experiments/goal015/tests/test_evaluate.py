"""Tiny synthetic GT and mock responses; no saved research outcomes or APIs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from offline_evaluation.experiments.goal015 import evaluate as e


def schedule(n=10):
    rows=[]
    for i in range(n):
        for builder in e.BUILDERS:
            for repeat in (1,2):
                block=f'scene{i}:{builder}:{repeat}'
                for condition in e.CONDITIONS:
                    rows.append({'call_id':block+':'+condition, 'scene_id':f'scene{i}',
                        'builder':builder,'condition':condition,'repeat_index':repeat,'paired_block_id':block})
    return rows


def observation(record, document=None, **updates):
    result={**record,'state':'SETTLED','sent':True,'response_received':True,
        'completed_response':True,'usage_known':True,'model_output_failure':False,'failure_kind':None,
        'response_text':json.dumps(document if document is not None else {'operations':[]})}
    result.update(updates)
    return result


def program(points):
    return {'operations':[{'op':'set','x':x,'y':y,'z':z,'block':material} for (x,y,z),material in points.items()]}


def tiny_gt():
    return e.core.Grid({(0,0,0):'stone',(1,0,0):'dirt',(0,1,0):'stone'})


def synthetic_rows(n=10):
    records=schedule(n);rows=[]
    for record in records:
        row=e._base(record,observation(record))
        row.update(protocol_valid=True,nonempty=True,empty_output=False,status='PROTOCOL_VALID',
                   evaluation_status='PASS',null_reason=None)
        value=.2 if record['condition']=='SUMMARY_ONLY' else .4
        row['metrics']={'values':{m:(1. if m=='aligned_dimension_mae' else value) for m in e.METRICS},'null_reasons':{}}
        rows.append(row)
    return records,rows


def test_tiny_gt_strict_translation_and_all_metric_parity():
    gt=tiny_gt();prediction=e.core.Grid({(x+12,y-5,z+7):m for (x,y,z),m in gt.voxels.items()})
    before=deepcopy((gt.voxels,prediction.voxels))
    result=e.evaluate_grids(gt,prediction)
    old=e.core.evaluate_pair(gt,prediction,alignment_mode='strict_translation')
    assert result['alignment']==old['alignment']
    assert result['alignment']['shift_xyz']==[-12,5,-7]
    assert result['values']['aligned_occupancy_iou']==old['aligned']['occupancy_iou']==1
    assert result['values']['aligned_material_aware_iou']==old['aligned']['material_aware_iou']==1
    assert result['values']['aligned_surface_f1_at_1']==1
    assert result['values']['aligned_dimension_mae']==0
    assert result['extended_provenance']['alignment_searched'] is False
    assert result['extended_provenance']['stored_alignment_shift_xyz']==[-12,5,-7]
    assert (gt.voxels,prediction.voxels)==before
    assert set(result['values'])==set(e.METRICS)


def test_exact_material_mismatch_and_same_occupancy_shift():
    gt=e.core.Grid({(0,0,0):'stone',(1,0,0):'stone'})
    prediction=e.core.Grid({(8,0,0):'dirt',(9,0,0):'dirt'})
    metrics=e.evaluate_grids(gt,prediction)
    assert metrics['values']['aligned_occupancy_iou']==1
    assert metrics['values']['aligned_material_aware_iou']==0
    assert metrics['values']['aligned_surface_f1_at_1']==1


def test_normal_empty_success_preserves_zero_and_undefined_precision():
    record=schedule(1)[0];row=e.evaluate_one(record,observation(record),tiny_gt())
    assert row['protocol_valid'] and row['empty_output'] and not row['nonempty']
    assert row['status']=='PROTOCOL_VALID' and row['failure_kind'] is None
    assert row['metrics']['values']['aligned_occupancy_iou']==0
    assert row['metrics']['values']['aligned_surface_f1_at_1']==0
    assert row['metrics']['values']['aligned_occupancy_precision'] is None
    assert e._value(row,'aligned_occupancy_precision','completed_attempt_utility')[0] is None
    assert not e.failure0_eligible(row)


@pytest.mark.parametrize('changes,expected',[
    ({'response_text':'not JSON'},'parse_failure'),
    ({'response_text':'[]'},'validation_failure'),
    ({'response_text':json.dumps({'operations':[{'op':'set','x':True,'y':0,'z':0,'block':'stone'}]})},'validation_failure'),
    ({'failure_kind':'truncation','model_output_failure':True,'completed_response':False},'truncation'),
    ({'failure_kind':'refusal','model_output_failure':True},'refusal'),
])
def test_only_identified_model_failures_have_bounded_utility_zero(changes,expected):
    record=schedule(1)[0]
    row=e.evaluate_one(record,observation(record,**changes),lambda:pytest.fail('GT read for failed output'))
    assert row['failure_kind']==expected and not row['protocol_valid']
    assert e._value(row,'aligned_occupancy_iou','completed_attempt_utility')==(0.,None,True)
    assert e._value(row,'aligned_surface_f1_at_1','common_valid_conditional_accuracy')[0] is None
    assert e._value(row,'aligned_dimension_mae','completed_attempt_utility')[0] is None


@pytest.mark.parametrize('changes',[
    {'state':'SENT_PENDING','usage_known':False},
    {'state':'UNKNOWN_USAGE','usage_known':False,'failure_kind':'truncation','model_output_failure':True},
    {'failure_kind':'response_identity_failure'},
    {'failure_kind':'auth_failure','response_received':True,'usage_known':False},
    {'failure_kind':'http_failure','http_status':520,'usage_known':False},
    {'failure_kind':'unknown_failure','model_output_failure':True},
])
def test_unknown_operational_and_identity_failures_are_null(changes):
    record=schedule(1)[0]
    row=e.evaluate_one(record,observation(record,**changes),lambda:pytest.fail('GT read for operational failure'))
    assert not e.failure0_eligible(row)
    assert e._value(row,'aligned_occupancy_iou','completed_attempt_utility')[0] is None


def test_gt_unavailable_is_not_model_failure_and_protocol_valid_retained():
    record=schedule(1)[0]
    row=e.evaluate_one(record,observation(record,program(tiny_gt().voxels)),None)
    assert row['protocol_valid'] and row['evaluation_status']=='NOT_COMPUTED'
    assert row['null_reason']=='GT_NOT_AVAILABLE'
    assert e._value(row,'aligned_occupancy_iou','completed_attempt_utility')[0] is None


def test_all_80_mock_rows_and_40_blocks_preserved_and_no_input_mutation():
    records,rows=synthetic_rows();before=deepcopy((records,rows))
    result=e.aggregate_rows(records,rows)
    assert (records,rows)==before
    assert result['planned_calls']==80 and result['planned_paired_blocks']==40
    assert result['planned_independent_buildings']==10
    assert len(result['primary_paired_estimates'])==8
    for row in result['primary_paired_estimates']:
        assert row['estimate']==pytest.approx(.2)
        assert row['eligible_paired_blocks']==20 and row['contributing_scenes']==10
        assert row['scenes_with_both_repeats']==10
        assert row['bootstrap_finite_draws']==2000 and row['ci95']==pytest.approx([.2,.2])
    assert all(row['protocol_valid_count']==20 for row in result['reliability'])


def test_known_failure_unknown_and_undefined_metric_denominators():
    records,rows=synthetic_rows()
    target=next(r for r in rows if r['builder']=='openai' and r['condition']=='FULL_RECORD' and r['scene_id']=='scene0' and r['repeat_index']==1)
    target.update(protocol_valid=False,status='MODEL_OUTPUT_FAILURE',failure_kind='validation_failure',
                  model_output_failure=True,metrics=None,null_reason='validation_failure')
    target=next(r for r in rows if r['builder']=='openai' and r['condition']=='FULL_RECORD' and r['scene_id']=='scene1' and r['repeat_index']==1)
    target.update(protocol_valid=False,state='UNKNOWN_USAGE',usage_known=False,metrics=None,null_reason='UNKNOWN_USAGE')
    target=next(r for r in rows if r['builder']=='openai' and r['condition']=='FULL_RECORD' and r['scene_id']=='scene2' and r['repeat_index']==1)
    target['metrics']['values']['aligned_surface_f1_at_1']=None
    target['metrics']['null_reasons']['aligned_surface_f1_at_1']='DEPENDENCY_UNAVAILABLE'
    result=e.aggregate_rows(records,rows)
    by={(r['builder'],r['metric'],r['estimand']):r for r in result['primary_paired_estimates']}
    assert by['openai','aligned_occupancy_iou',e.POLICIES[0]]['eligible_paired_blocks']==19
    assert by['openai','aligned_occupancy_iou',e.POLICIES[1]]['eligible_paired_blocks']==18
    assert by['openai','aligned_surface_f1_at_1',e.POLICIES[0]]['eligible_paired_blocks']==18
    assert by['openai','aligned_surface_f1_at_1',e.POLICIES[1]]['eligible_paired_blocks']==17


def test_repeat_then_scene_weighting_and_duplicate_scene_bootstrap():
    records,rows=synthetic_rows(3)
    for row in rows:
        if row['builder']=='openai':
            if row['scene_id']=='scene2' or (row['scene_id']=='scene1' and row['repeat_index']==2):
                row.update(protocol_valid=False,state='NOT_SENT',metrics=None,null_reason='NOT_SENT')
            else:
                value=0. if row['condition']=='SUMMARY_ONLY' else (0. if row['scene_id']=='scene0' else .9)
                row['metrics']['values']['aligned_occupancy_iou']=value
    draws=np.tile([0,1,1],(2000,1))
    result=e._paired(rows,['scene0','scene1','scene2'],'openai','aligned_occupancy_iou',e.POLICIES[1],draws)
    assert result['estimate']==pytest.approx(.45)  # Per-pair pooling would be .3.
    assert result['bootstrap_draws'][0]==pytest.approx(.6)
    assert result['planned_scenes']==3 and result['contributing_scenes']==2
    assert result['scenes_with_both_repeats']==1 and result['scenes_with_one_repeat']==1 and result['scenes_with_no_repeats']==1


@pytest.mark.parametrize('finite,scenes,point,passes',[(1900,2,.1,True),(1899,2,.1,False),(2000,1,.1,False),(2000,2,None,False)])
def test_ci_gate_without_draw_replacement(finite,scenes,point,passes):
    values=[float(i) for i in range(finite)]+[None]*(2000-finite)
    result=e._interval(point,values,scenes)
    assert (result['ci95'] is not None)==passes
    assert result['bootstrap_finite_draws']==finite and result['bootstrap_undefined_draws']==2000-finite


def test_no_ledger_is_complete_planned_null_frame_without_gt_reads(tmp_path):
    result=e.evaluate_saved(schedule(),[],tmp_path/'evaluation',lambda _:pytest.fail('unsent GT read'),fixture=True)
    assert result['planned_calls']==80 and result['actual_metric_rows']==0
    assert all(r['estimate'] is None and r['ci95'] is None for r in result['primary_paired_estimates'])
    data=json.loads((tmp_path/'evaluation/evaluation_80.json').read_bytes())
    assert len(data['rows'])==80 and all(r['status']=='NOT_SENT' for r in data['rows'])


def descriptor(path,data):
    path.write_bytes(data)
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def test_saved_raw_text_hashes_parsed_voxel_metrics_linked_and_exclusive(tmp_path):
    records=schedule(1);saved=[]
    for i,r in enumerate(records):
        row=observation(r,program(tiny_gt().voxels))
        row['raw']=descriptor(tmp_path/f'raw{i}.json',b'{"fixture":true}')
        row['response_text_path']=descriptor(tmp_path/f'text{i}.txt',row.pop('response_text').encode())
        saved.append(row)
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    result=e.evaluate_saved(records,saved,tmp_path/'eval',lambda _:tiny_gt(),fixture=True)
    assert result['actual_metric_rows']==8
    data=json.loads((tmp_path/'eval/evaluation_80.json').read_bytes())
    for row in data['rows']:
        assert set(row['artifacts'])=={'raw','response_text_path','parsed.json','voxels.json','metrics.json'}
        for desc in row['artifacts'].values():assert hashlib.sha256(Path(desc['path']).read_bytes()).hexdigest()==desc['sha256']
    assert all(p.read_bytes()==raw for p,raw in before.items())
    with pytest.raises(FileExistsError):e.evaluate_saved(records,saved,tmp_path/'eval',lambda _:tiny_gt(),fixture=True)
    Path(saved[0]['raw']['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'):
        e.evaluate_saved(records,saved,tmp_path/'tampered-eval',lambda _:tiny_gt(),fixture=True)


def test_import_never_imports_provider_or_transport_module():
    code='import sys; from offline_evaluation.experiments.goal015 import evaluate as e; e.frozen_grid_sha256(e.core.Grid({})); assert "offline_evaluation.server_v2.providers" not in sys.modules; assert "offline_evaluation.server_v2.generation" not in sys.modules'
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def test_schedule_duplicate_missing_or_cross_scene_block_rejected():
    records=schedule();bad=deepcopy(records);bad.pop()
    with pytest.raises(ValueError):e.aggregate_rows(bad,[])
    bad=deepcopy(records);bad[0]['paired_block_id']=bad[8]['paired_block_id']
    with pytest.raises(ValueError):e.aggregate_rows(bad,[])


def test_runtime_send_status_preserves_uncertainty_without_assuming_post():
    record=schedule(1)[0]
    complete=observation(record,response_text='invalid JSON')
    complete.pop('sent');complete['send_status']='POST_RETURNED'
    row=e.evaluate_one(record,complete)
    assert row['send_started'] and row['sent'] and e.failure0_eligible(row)
    uncertain=observation(record,state='SENT_PENDING',usage_known=False,response_received=False,
                           completed_response=False,send_status='POSSIBLY_SENT')
    uncertain.pop('sent')
    row=e.evaluate_one(record,uncertain)
    assert row['send_started'] and not row['sent'] and not e.failure0_eligible(row)


def test_local_execution_memory_failure_is_not_model_failure(monkeypatch):
    record=schedule(1)[0]
    monkeypatch.setattr(e.core,'execute',lambda *args:(_ for _ in ()).throw(MemoryError()))
    row=e.evaluate_one(record,observation(record),tiny_gt())
    assert row['status']=='LOCAL_RESOURCE_FAILURE' and not e.failure0_eligible(row)
    assert e._value(row,'aligned_occupancy_iou',e.POLICIES[0])[0] is None


def test_mock80_saved_response_roundtrip_normal_empty_failure_unknown(tmp_path):
    records=schedule();saved=[]
    for i,record in enumerate(records):
        row=observation(record)
        row.pop('sent');row['send_status']='POST_RETURNED'
        if i==1:row['response_text']='not JSON'
        if i==9:row.update(state='UNKNOWN_USAGE',usage_known=False)
        row['raw']=descriptor(tmp_path/f'r{i}.json',json.dumps({'fixture_call':i}).encode())
        row['response_text_path']=descriptor(tmp_path/f't{i}.txt',row.pop('response_text').encode())
        saved.append(row)
    summary=e.evaluate_saved(records,saved,tmp_path/'eval80',lambda _:tiny_gt(),fixture=True)
    assert summary['planned_calls']==80 and summary['actual_metric_rows']==78
    evaluated=json.loads((tmp_path/'eval80/evaluation_80.json').read_bytes())['rows']
    assert len(evaluated)==80
    assert sum(row['protocol_valid'] and row['empty_output'] for row in evaluated)==78
    assert evaluated[1]['failure_kind']=='parse_failure' and e.failure0_eligible(evaluated[1])
    assert evaluated[9]['status']=='OPERATIONAL_UNKNOWN' and not e.failure0_eligible(evaluated[9])
    assert sum(row['sent_count'] for row in summary['reliability'])==80


def test_missing_saved_raw_rejected_even_with_valid_text(tmp_path):
    record=schedule(1)[0];row=observation(record)
    row['response_text_path']=descriptor(tmp_path/'text.txt',row.pop('response_text').encode())
    with pytest.raises(ValueError,match='saved raw'):
        e.evaluate_saved(schedule(1),[row],tmp_path/'out',lambda _:tiny_gt(),fixture=True)


def test_disabled_real_cli_writes_only_evaluation_no_gt_or_api_module(tmp_path):
    from offline_evaluation.experiments.goal015 import runtime
    root=tmp_path/'canonical';root.mkdir()
    payload=descriptor(root/'payload.json',b'{"fixture":true}')
    spec=descriptor(root/'spec.md',b'fixture specification')
    pricing=descriptor(root/'pricing.json',b'{"fixture":true}')
    code=descriptor(root/'fixture_code.py',b'# fixture code')
    records=schedule()
    for i,row in enumerate(records):row.update(call_id=f'c{i:03}',payload=payload,reserve_usd='0.01',input_reservation_tokens=100)
    plan={'schema':'goal015_disabled_plan_v1','canonical_root':str(root),'api_allowed':False,'budget_approved':False,
        'fixture_only':True,'proposed_budget_usd':'5','planned_calls':80,'planned_buildings':10,
        'concurrency':1,'max_retries':0,'timeout_seconds':600,'pricing_contract':{'freshness_seconds':86400},'pricing_evidence':pricing,
        'implementation_hashes':{code['path']:code['sha256']},'specification':spec,'schedule':records,
        'gt_sources':{scene:{'building_directory':str(root/'nonexistent'/scene),'bbox_sha256':'a'*64,
                    'voxels_sha256':'b'*64,'grid_sha256':'c'*64} for scene in {r['scene_id'] for r in records}}}
    plan['plan_id']=runtime.identity(plan);path=root/'plan.json';path.write_text(json.dumps(plan))
    raw_before=path.read_bytes()
    result=subprocess.run([sys.executable,'-B','-m','offline_evaluation.experiments.goal015.evaluate',
                           '--plan',str(path),'--output-dir',str(root/'evaluation')],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['actual_metric_rows']==0
    assert path.read_bytes()==raw_before
    assert not (root/'enabled').exists() and not (root/'execution.lock').exists()
    assert not (root/'authority.json').exists()
    product=json.loads((root/'evaluation/evaluation_80.json').read_bytes())
    assert product['plan_binding']['plan_id']==plan['plan_id']
    assert len(product['rows'])==80 and all(r['artifacts']['request_payload']==payload for r in product['rows'])


def test_original_schema_gt_loader_uses_frozen_hash_contract_and_rejects_tampering(tmp_path,monkeypatch):
    from offline_evaluation.experiments.goal015 import runtime
    root=tmp_path/'canonical';root.mkdir()
    building=root/'fixture/building_000';(building/'gt').mkdir(parents=True)
    voxels=building/'gt/voxels.npy';bbox=building/'gt/bbox.json'
    np.save(voxels,np.array([[['stone']],[['dirt']]],dtype='<U5'),allow_pickle=False)
    bbox.write_text(json.dumps({'xmin':0,'xmax':0,'ymin':2,'ymax':3,'zmin':0,'zmax':0,'voxel_axis_order':'Y,X,Z'}))
    truth=e.core.Grid({(0,2,0):'stone',(0,3,0):'dirt'},origin=(0,2,0))
    # Original schema and its exact frozen object_hash helper, not an assumed
    # compact JSON or alternate grid serialization.
    original={'building_directory':str(building),'voxels_sha256':runtime.descriptor(voxels)['sha256'],
              'bbox_sha256':runtime.descriptor(bbox)['sha256'],
              'grid_sha256':e._frozen_object_hash()(e.core.grid_to_payload(truth))}
    records=schedule()
    for record in records:
        if record['scene_id']=='scene0':record['scene_id']='fixture_building_000'
    sources={record['scene_id']:original for record in records}
    plan={'canonical_root':str(root),'fixture_only':True,'plan_id':'fixtureplan','schedule':records,'gt_sources':sources}
    path=root/'plan.json';path.write_text(json.dumps(plan))
    row=observation(records[0],program(truth.voxels))
    row['raw']=descriptor(root/'response.json',b'{"fixture":true}')
    row['response_text_path']=descriptor(root/'response.txt',row.pop('response_text').encode())
    (root/'enabled').mkdir();(root/'execution.lock').write_bytes(b'')
    (root/'enabled/ledger.json').write_text(json.dumps({'calls':[row]}))
    (root/'enabled/approval.json').write_text('{}')
    monkeypatch.setattr(runtime,'verify_plan',lambda _:plan)
    monkeypatch.setattr(runtime,'_validate_ledger',lambda *args:None)
    result=e.evaluate_plan(path,root/'evaluation')
    assert result['actual_metric_rows']==1
    artifact=json.loads(Path(result['evaluation_artifact']['path']).read_bytes())
    measured=artifact['rows'][0]
    assert measured['metrics']['gt_grid_sha256']==original['grid_sha256']
    assert measured['metrics']['values']['aligned_occupancy_iou']==1
    assert measured['gt_source']==original
    bbox.write_text('{}')
    with pytest.raises(ValueError,match='original GT bytes differ'):
        e.evaluate_plan(path,root/'tampered_evaluation')
