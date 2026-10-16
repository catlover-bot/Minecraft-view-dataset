"""Synthetic-only controls, exact material arithmetic and immutable bindings."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from offline_evaluation.analysis.goal010 import exact_oracle as e
from offline_evaluation.server_v2.core import Grid, Limits, ValidationError, evaluate_pair, grid_to_payload, metrics
from offline_evaluation.server_v2.data import load_scene


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(e.json_bytes(value))


def descriptor(path, root=None):
    return {'path':str(path.relative_to(root)) if root else str(path), 'sha256':e.sha256_file(path),'bytes':path.stat().st_size}


def gate(digest='a'*64):
    return {'status':'PASS','input_kind':'fixture','phase0_binding':'PASS','definitions_frozen':True,
            'specification_sha256':'b'*64,'code_commit':'c'*40,'goal007_manifest_sha256':digest}


@pytest.fixture
def fixture(tmp_path):
    schedule, scenes, gt_sources = [], {}, {}
    for index in (34, 99):
        directory=tmp_path/'buildings_100_v4'/f'building_{index:03d}'
        (directory/'gt').mkdir(parents=True)
        # Padding and asymmetric dimensions expose YXZ/XYZ and origin mistakes.
        array=np.full((3,4,5),'air',dtype='<U20')
        array[1,1:3,1:4]='brick_block';array[2,2,3]='stone_slab'
        np.save(directory/'gt/voxels.npy',array,allow_pickle=False)
        write(directory/'gt/bbox.json',{'xmin':-5,'xmax':-2,'ymin':7,'ymax':9,'zmin':-10,'zmax':-6,'voxel_axis_order':'Y,X,Z'})
        scene=load_scene(directory);scenes[scene.scene_id]=scene
        gt_sources[scene.scene_id]={'building_directory':str(directory),'bbox_sha256':e.sha256_file(directory/'gt/bbox.json'),
            'voxels_sha256':e.sha256_file(directory/'gt/voxels.npy'),'grid_sha256':e.object_hash(scene.grid.to_dict())}
        text=e.cuboids.exact_builder_text(e.cuboids.encode_exact(scene.grid))
        for builder in e.BUILDERS:
            schedule.append({'schedule_index':len(schedule),'provider':builder,'scene_id':scene.scene_id,'condition':'E',
                'input_id':scene.scene_id+'-E','input_text':text,'input_sha256':hashlib.sha256(text.encode()).hexdigest()})
    enabled=tmp_path/'enabled';enabled.mkdir()
    identity={'schedule':schedule,'gt_by_scene':gt_sources,'enabled_directory':str(enabled),'limits':asdict(Limits()),'budgets':{'max_output_tokens':32768}}
    plan={'identity':identity,'identity_hash':e.object_hash(identity)};write(tmp_path/'plan.json',plan)
    entries,slots=[],[]
    for record in schedule:
        index=record['schedule_index'];scene=scenes[record['scene_id']]
        success=index!=2;attempt='fixture'+str(index)
        artifacts={}
        raw=enabled/'attempts'/attempt/'raw.json';write(raw,{'fixture':True});artifacts['raw.json']=descriptor(raw,enabled)
        operations=None;prediction=None
        if success:
            prediction=Grid({(x+(1 if index==3 else 0),y,z):('stonebrick' if index==0 and material=='brick_block' else material)
                for (x,y,z),material in scene.grid.voxels.items()})
            operations=e.cuboids.reference_actions(e.cuboids.encode_exact(prediction))['operations']
            for name,payload in [('parsed.json',{'operations':operations}),('voxels.json',grid_to_payload(prediction))]:
                path=enabled/'attempts'/attempt/name;write(path,payload);artifacts[name]=descriptor(path,enabled)
        usage={'input_tokens':100,'output_tokens':50 if success else 32768,'output_tokens_details':{'reasoning_tokens':5} if record['provider']=='openai' else {}}
        entry={'attempt_id':attempt,'schedule_index':index,'provider':record['provider'],'input_id':record['input_id'],
            'status':'success' if success else 'failed','usage':usage,'artifacts':artifacts}
        entries.append(entry)
        slots.append({'scene_id':record['scene_id'],'provider':record['provider'],'condition':'E','input_id':record['input_id'],
            'input_sha256':record['input_sha256'],'global_schedule_index':index,'attempt_id':attempt,
            'source_label':'fixture','source_run':str(enabled),'source_plan_path':str(tmp_path/'plan.json'),
            'source_plan_identity':plan['identity_hash'],'source_entry_hash':e.object_hash(entry),
            'source_gt_descriptor_hash':e.object_hash(gt_sources[scene.scene_id]),'gt_grid_sha256':gt_sources[scene.scene_id]['grid_sha256'],
            'generation_status':entry['status'],'status':entry['status'],'usage':usage,'usage_known':True,'completed_request_outcome':True,
            'artifacts':artifacts,'metrics':evaluate_pair(scene.grid,prediction,alignment_mode='strict_translation') if success else None,
            'operation_count':len(operations) if success else None,'final_voxel_count':len(prediction.voxels) if success else None,
            'failure_kind':None if success else 'truncation','natural_stop':success})
    ledger={'plan_identity_hash':plan['identity_hash'],'entries':entries};write(enabled/'ledger.json',ledger)
    fixed={'identity':{'full_schedule':schedule}};fixed['identity_hash']=e.object_hash(fixed['identity']);write(tmp_path/'full_manifest.json',fixed)
    old=tmp_path/'old';write(old/'data/final_slots_960.json',slots)
    source={'plan_path':str(tmp_path/'plan.json'),'plan_sha256':e.sha256_file(tmp_path/'plan.json'),
        'ledger_path':str(enabled/'ledger.json'),'ledger_sha256':e.sha256_file(enabled/'ledger.json'),
        'enabled_directory':str(enabled),'plan_identity_hash':plan['identity_hash']}
    manifest={'products':{'data/final_slots_960.json':descriptor(old/'data/final_slots_960.json',old)},
        'source_manifest':{**descriptor(tmp_path/'full_manifest.json'),'identity_hash':fixed['identity_hash']},'sources':{'fixture':source}}
    write(old/'MANIFEST.json',manifest)
    return {'root':tmp_path,'slots':slots,'scenes':scenes,'manifest':old/'MANIFEST.json','gate':gate(e.sha256_file(old/'MANIFEST.json'))}


def test_exact_material_iou_uses_coordinate_material_union():
    gt={(0,0,0):'brick_block',(1,0,0):'stone_slab'}
    pred={(0,0,0):'stonebrick',(1,0,0):'stone_slab'}
    values,confusion=e.independent_metrics(gt,pred)
    assert values['occupancy_iou']==1 and values['dimension_mae']==0
    assert values['material_aware_iou']==1/3  # not the incorrect1/2 occupancy-union denominator.
    assert values['material_set_f1']==.5 and values['replace_count']==1
    assert confusion['brick_block','stonebrick']==1


def test_material_set_identity_can_hide_complete_spatial_swap():
    gt={(0,0,0):'a',(1,0,0):'b'};pred={(0,0,0):'b',(1,0,0):'a'}
    result,_=e.independent_metrics(gt,pred)
    assert result['material_set_f1']==result['occupancy_iou']==1
    assert result['material_aware_iou']==0 and result['replace_count']==2


@pytest.mark.parametrize('g,p', [({},{}), ({(0,0,0):'a'},{}), ({},{(-1,2,3):'a'}),
    ({(0,0,0):'a',(1,0,0):'b'},{(0,0,0):'c',(4,0,0):'b'})])
def test_independent_confusion_agrees_with_frozen_formula(g,p):
    actual,pairs=e.independent_metrics(g,p)
    assert actual==metrics(Grid(g),Grid(p))
    assert sum(pairs.values())==len(set(g)|set(p))


def test_alias_is_static_narrow_and_never_replaces_strict_values():
    gt={(0,0,0):'brick_block',(1,0,0):'stone_slab',(2,0,0):'stonebrick'}
    pred={(0,0,0):'bricks',(1,0,0):'double_stone_slab',(2,0,0):'stone_bricks'}
    before=deepcopy((gt,pred));strict,_=e.independent_metrics(gt,pred);alias=e.lexical_sensitivity(gt,pred)
    assert strict['material_aware_iou']==0 and alias['material_aware_iou']==.5
    assert (gt,pred)==before
    assert e.lexical_sensitivity({(0,0,0):'brick_block'},{(0,0,0):'stonebrick'})['material_aware_iou']==0
    assert e.sha256_file(e.ALIAS_SOURCE)==e.ALIAS_SOURCE_SHA256


def test_missing_gate_stops_before_iterating_data(tmp_path):
    class Forbidden:
        def __iter__(self):raise AssertionError('actual inputs accessed before gate')
    with pytest.raises(ValueError,match='Phase0 binding'):
        e.run_exact_oracle(Forbidden(),tmp_path/'missing',tmp_path/'output',gate={'status':'PASS'},specification_sha256='b'*64)
    assert not (tmp_path/'output').exists()


def test_reference_control_preserves_padding_negative_origin_and_yxz(fixture):
    scene=next(iter(fixture['scenes'].values()))
    text=e.cuboids.exact_builder_text(e.cuboids.encode_exact(scene.grid))
    spec,control=e.reference_control(scene,text,Limits())
    assert spec['schema']=='resmoke_v2_exact_cuboids_v1'
    assert control['storage_dimensions_xyz']==[4,3,5]
    assert control['occupied_dimensions_xyz']==[2,2,3]
    assert control['origin_xyz']==[-5,7,-10]
    assert scene.grid.voxels[-4,8,-9]=='brick_block'
    assert control['reference_execution_exact'] is True
    assert control['control_is_model_success'] is False
    assert control['reference_tokens'] is None


def test_changed_fixed_palette_or_prefix_fails_control(fixture):
    scene=next(iter(fixture['scenes'].values()))
    text=e.cuboids.exact_builder_text(e.cuboids.encode_exact(scene.grid))
    with pytest.raises(ValueError,match='immutable E input'):
        e.reference_control(scene,text.replace('brick_block','stonebrick'),Limits())
    with pytest.raises(ValueError,match='immutable E input'):
        e.reference_control(scene,'different preamble\n'+text.split('\n',1)[1],Limits())


def test_reference_expansion_limit_fails_without_partial_output(fixture):
    scene=next(iter(fixture['scenes'].values()))
    text=e.cuboids.exact_builder_text(e.cuboids.encode_exact(scene.grid))
    with pytest.raises(ValidationError,match='voxel limit'):
        e.reference_control(scene,text,replace(Limits(),max_voxels=2))


def test_exact_operation_boundary_wrong_material_is_counted():
    spec=e.cuboids.encode_exact(Grid({(0,0,0):'brick_block',(1,0,0):'brick_block'}))
    operations=e.cuboids.reference_actions(spec)['operations'];operations[0]['block']='stonebrick'
    value=e.operation_transcription(spec,operations)
    assert value['reference_boundaries_seen']==1 and value['reference_boundaries_with_correct_label']==0
    assert value['boundary_label_substitutions']==[{'expected_material':'brick_block','operation_material':'stonebrick','operation_count':1}]
    assert value['touched_voxel_visits']==2


def test_equivalent_partition_is_not_called_voxel_omission():
    spec=e.cuboids.encode_exact(Grid({(0,0,0):'stone',(1,0,0):'stone'}))
    operations=[{'op':'set','x':x,'y':0,'z':0,'block':'stone'} for x in (0,1)]
    value=e.operation_transcription(spec,operations)
    assert value['reference_boundaries_without_literal_match']==1
    assert 'no omission' in value['caveat']
    assert e.execute(operations).voxels==e.cuboids.decode_exact(spec).voxels


def test_strict_shift_certificate_required_and_not_refitted(fixture):
    packages,_=e.bind_sources(fixture['slots'],fixture['manifest'],gate=fixture['gate'])
    package=next(p for p in packages if p['slot']['global_schedule_index']==3)
    scene=fixture['scenes'][package['slot']['scene_id']]
    spec,control=e.reference_control(scene,package['record']['input_text'],Limits())
    outcome,_,_=e.audit_outcome(package,scene,spec,control)
    assert outcome['stored_alignment_shift_xyz']==[-1,0,0]
    assert outcome['independent_metrics']['aligned']['occupancy_iou']==1
    assert outcome['independent_metrics']['absolute']['occupancy_iou']<1
    bad=deepcopy(package);bad['slot']['metrics']['alignment']['search_complete']=False
    with pytest.raises(ValueError,match='certificate'):e.audit_outcome(bad,scene,spec,control)


def test_unknown_outcome_is_null_never_zero_or_reference_success(fixture):
    packages,_=e.bind_sources(fixture['slots'],fixture['manifest'],gate=fixture['gate'])
    p=deepcopy(next(p for p in packages if p['slot']['status']=='failed'))
    p['slot'].update(status='operational_unmeasurable',usage=None,usage_known=False,completed_request_outcome=False,failure_kind='http_failure')
    scene=fixture['scenes'][p['slot']['scene_id']];spec,control=e.reference_control(scene,p['record']['input_text'],Limits())
    outcome,capacity,confusion=e.audit_outcome(p,scene,spec,control)
    assert outcome['outcome_category']=='other' and outcome['independent_metrics'] is None
    assert capacity['output_tokens_observed'] is None and capacity['completion_counterfactual'] is None and confusion==[]


def test_artifact_tamper_and_path_escape_fail_closed(fixture):
    row=fixture['slots'][0];path=Path(row['source_run'])/row['artifacts']['voxels.json']['path']
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='hash/size'):e.bind_sources(fixture['slots'],fixture['manifest'],gate=fixture['gate'])
    with pytest.raises(ValueError,match='traversal'):e._artifact(fixture['root'],{'path':'../outside','sha256':'a'*64,'bytes':0},{})


def test_duplicate_and_changed_selected_slot_binding_rejected(fixture):
    slots=deepcopy(fixture['slots']);slots[0]['attempt_id']='changed'
    with pytest.raises(ValueError,match='selected slots differ'):e.bind_sources(slots,fixture['manifest'],gate=fixture['gate'])


def test_bad_cached_metric_is_not_silently_replaced(fixture):
    packages,_=e.bind_sources(fixture['slots'],fixture['manifest'],gate=fixture['gate'])
    p=packages[0];p['slot']['metrics']['aligned']['material_aware_iou']=.9
    scene=fixture['scenes'][p['slot']['scene_id']];spec,control=e.reference_control(scene,p['record']['input_text'],Limits())
    with pytest.raises(ValueError,match='metric disagreement'):e.audit_outcome(p,scene,spec,control)


def test_full_fixture_completion_artifacts_and_original_sources(fixture):
    before={str(p):e.sha256_file(p) for p in fixture['root'].rglob('*') if p.is_file()}
    out=fixture['root']/'new_audit'
    result=e.run_exact_oracle(fixture['slots'],fixture['manifest'],out,gate=fixture['gate'],specification_sha256='b'*64)
    assert result['status']=='E_AUDIT_PASS' and result['outcomes']==4 and result['controls']==2
    assert result['valid_strict_metric_checks']==3 and result['research_api_requests_sent']==0
    assert result['outcome_counts']=={'openai':{'success':1,'truncation':1,'validation':0,'other':0},'anthropic':{'success':2,'truncation':0,'validation':0,'other':0}}
    assert before=={p:e.sha256_file(p) for p in before}
    for name,metadata in result['products'].items():assert e.sha256_file(out/name)==metadata['sha256']
    taxonomy=e.read_json(out/'E_error_taxonomy.json')
    assert taxonomy['strict_metrics_changed'] is False
    assert taxonomy['outcomes'][0]['near_perfect_occupancy_low_material'] is False # Anthropic sorted before OpenAI.
    assert any(r['near_perfect_occupancy_low_material'] for r in taxonomy['outcomes'])
    assert all(r['reference_tokens'] is None for r in e.read_json(out/'E_reference_controls.json'))
    with pytest.raises(ValueError,match='new E diagnostic'):e.run_exact_oracle(fixture['slots'],fixture['manifest'],out,gate=fixture['gate'],specification_sha256='b'*64)


def test_expected_specification_mismatch_stops_before_source_reads(fixture):
    with pytest.raises(ValueError,match='specification SHA mismatch'):
        e.run_exact_oracle(None,'missing',fixture['root']/'unused',gate=fixture['gate'],specification_sha256='d'*64)
    assert not (fixture['root']/'unused').exists()


def test_bool_metric_count_is_not_accepted_as_integer():
    actual,_=e.independent_metrics({(0,0,0):'stone'},{(0,0,0):'stone'})
    expected=deepcopy(actual);expected['material_match_count']=True
    with pytest.raises(ValueError,match='metric disagreement'):e.verify_metrics(actual,expected)


def test_unknown_tokens_never_turn_truncation_label_into_known_failure(fixture):
    packages,_=e.bind_sources(fixture['slots'],fixture['manifest'],gate=fixture['gate'])
    package=next(p for p in packages if p['slot']['status']=='failed')
    package['slot'].update(usage=None,usage_known=False,completed_request_outcome=False)
    scene=fixture['scenes'][package['slot']['scene_id']]
    spec,control=e.reference_control(scene,package['record']['input_text'],Limits())
    outcome,capacity,_=e.audit_outcome(package,scene,spec,control)
    assert outcome['outcome_category']=='other' and capacity['output_cap_fraction_observed'] is None
    summaries=e.capacity_summary([capacity])
    assert sum(r['outcomes'] for r in summaries if r['outcome_category']=='truncation')==0
    assert sum(r['outcomes'] for r in summaries if r['outcome_category']=='other')==1


def test_duplicate_voxel_coordinates_rejected_by_source_reader(fixture):
    packages,_=e.bind_sources(fixture['slots'],fixture['manifest'],gate=fixture['gate'])
    package=next(p for p in packages if p['slot']['status']=='success')
    original=e.read_json(package['artifacts']['voxels.json']);original['voxels'].append(original['voxels'][0])
    path=fixture['root']/'duplicate_voxels.json';write(path,original)
    package['artifacts']['voxels.json']=str(path)
    scene=fixture['scenes'][package['slot']['scene_id']]
    spec,control=e.reference_control(scene,package['record']['input_text'],Limits())
    with pytest.raises(ValidationError,match='duplicate voxel'):e.audit_outcome(package,scene,spec,control)
