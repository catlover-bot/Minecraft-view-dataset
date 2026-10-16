"""Synthetic baseline boundaries, persistence, matching and cluster inference."""
from collections import Counter,defaultdict
from copy import deepcopy
from decimal import Decimal
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from offline_evaluation.analysis.goal010 import baselines as base
from offline_evaluation.analysis.goal010 import independent_geometry as independent
from offline_evaluation.analysis.goal010.tests.test_factors import make_case
from offline_evaluation.server_v2.core import Grid,Limits,ValidationError,execute


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*a,**k):pytest.fail('network forbidden in baseline fixtures')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket.socket,'connect_ex',denied)


def bbox(x=3,y=3,z=3,origin=(0,0,0)):
    return {a+suffix:origin[i]+(0 if suffix=='min' else (x,y,z)[i]-1)
            for i,a in enumerate('xyz') for suffix in ('min','max')}


def construction_records():
    return [{'scene_id':'fixture-scene','condition':'O','description_model':'openai','input_id':'fixture-O',
             'input_sha256':'a'*64,'source_sha256':'b'*64,'permitted_input':{'width':3,'height':2,'depth':4}},
            {'scene_id':'fixture-scene','condition':'P','description_model':None,'input_id':'fixture-P',
             'input_sha256':'c'*64,'permitted_input':bbox(3,2,4,(-2,1,-4))}]


def test_leakage_boundary_signatures_do_not_accept_gt(monkeypatch):
    assert list(inspect.signature(base.original_actions).parameters)==['dimensions_estimate','structure','limits']
    assert list(inspect.signature(base.partial_actions).parameters)==['world_bbox','structure','limits']
    def forbidden(*a,**k):pytest.fail('GT accessed during baseline construction')
    monkeypatch.setattr(base,'load_scene',forbidden);monkeypatch.setattr(base,'evaluate_pair',forbidden)
    dims={'width':3,'depth':4,'height':2};box=bbox(3,2,4,(-10,8,2))
    assert execute(base.original_actions(dims,'filled_box')).bbox()==bbox(3,2,4)
    assert execute(base.partial_actions(box,'filled_box')).bbox()==box
    with pytest.raises(TypeError):base.original_actions(dims,'filled_box',gt_bbox=box)
    with pytest.raises(TypeError):base.partial_actions(box,'filled_box',voxel_count=5)
    richer={**dims,'world_bbox':box,'voxel_count':999,'materials':['gold_block'],'style':'forbidden influence'}
    assert base.original_actions(richer,'filled_box')==base.original_actions(dims,'filled_box')


@pytest.mark.parametrize('value',[None,True,False,0,-1,float('nan'),float('inf'),Decimal('NaN'),'4',10**1000,Decimal('sNaN')])
def test_original_invalid_dimensions_never_filled_from_gt(value):
    with pytest.raises(ValidationError):base.original_actions({'width':value,'height':3,'depth':4},'filled_box')


def test_missing_dimensions_and_round_to_even_axis_mapping():
    for value in (None,{}, {'width':3,'depth':4}):
        with pytest.raises(ValidationError):base.original_actions(value,'filled_box')
    grid=execute(base.original_actions({'width':2.5,'height':3.5,'depth':.2},'filled_box'))
    assert grid.dimensions()==(2,4,1) and grid.bbox()==bbox(2,4,1)
    assert set(grid.voxels.values())=={'stone'}


@pytest.mark.parametrize('shape',[(1,1,1),(1,3,4),(2,2,2),(2,3,4),(3,3,3),(4,5,6)])
def test_inclusive_negative_box_and_shell_unique_voxels(shape):
    bounds=bbox(*shape,origin=(-3,7,-2));lo=(-3,7,-2)
    filled=set(itertools.product(*[range(lo[i],lo[i]+shape[i]) for i in range(3)]))
    shell={c for c in filled if any(c[i] in (lo[i],lo[i]+shape[i]-1) for i in range(3))}
    for structure,expected in (('filled_box',filled),('six_face_shell',shell)):
        actual=execute(base.partial_actions(bounds,structure))
        assert set(actual.voxels)==expected and len(actual.voxels)==len(expected)
        assert actual.bbox()==bounds


@pytest.mark.parametrize('mutation',['missing','reversed','bool','float','coordinate_limit','unsupported_shape'])
def test_invalid_partial_bounds_and_resource_limits_fail_closed(mutation):
    bounds=bbox();structure='filled_box';limits=Limits()
    if mutation=='missing':bounds.pop('xmin')
    elif mutation=='reversed':bounds['xmin']=bounds['xmax']+1
    elif mutation=='bool':bounds['xmin']=False
    elif mutation=='float':bounds['xmin']=0.0
    elif mutation=='coordinate_limit':limits=Limits(max_coordinate_abs=1)
    else:structure='tuned_to_gt'
    with pytest.raises(ValidationError):base.partial_actions(bounds,structure,limits)


def test_generation_freezes_every_prediction_before_gt_evaluation(tmp_path,monkeypatch):
    def forbidden(*a,**k):pytest.fail('GT/evaluation accessed before all predictions persisted')
    monkeypatch.setattr(base,'load_scene',forbidden);monkeypatch.setattr(base,'evaluate_pair',forbidden)
    records=construction_records();before=deepcopy(records)
    manifest=base.generate_predictions(records,tmp_path/'predictions')
    assert records==before and len(manifest['predictions'])==4
    assert manifest['counts']=={'PREDICTION_SAVED':4}
    assert (tmp_path/'baseline_predictions_manifest.json').is_file()
    assert manifest['GT_voxels_read_during_construction'] is False and manifest['Builder_replication_of_predictions'] is False
    for row in manifest['predictions']:
        assert set(row['artifacts'])=={'actions','coordinates'}
        for artifact in row['artifacts'].values():
            p=tmp_path/'predictions'/artifact['path'];assert p.is_file()
            assert base.sha256_file(p)==artifact['sha256'] and p.stat().st_size==artifact['bytes']
    gt=Grid({(-2,1,-4):'brick_block',(-1,1,-4):'stone_slab'})
    scene=SimpleNamespace(scene_id='fixture-scene',grid=gt,bbox=bbox(3,2,4,(-2,1,-4)))
    def load(*a,**k):
        assert json.loads((tmp_path/'baseline_predictions_manifest.json').read_text())['status']=='ALL_PREDICTIONS_FROZEN_BEFORE_GT_EVALUATION'
        assert len(list((tmp_path/'predictions').glob('*.xyz.npy')))==4
        return scene
    monkeypatch.setattr(base,'load_scene',load)
    from offline_evaluation.server_v2.core import evaluate_pair
    monkeypatch.setattr(base,'evaluate_pair',evaluate_pair)
    package={'descriptions':[{'scene_id':'fixture-scene','gt_descriptor':{'building_directory':'fixture-only',
        'grid_sha256':base.object_hash(gt.to_dict()),'bbox_sha256':'d'*64,'voxels_sha256':'e'*64}}],
        'cohort':{'scenes':[{'scene_id':'fixture-scene','size_rank_stratum':0}]}}
    evaluated=base.evaluate_predictions(manifest['predictions'],package,tmp_path/'predictions',Limits())
    assert all(r['status']=='success' and r['independent_check_status']=='PASS' for r in evaluated)
    p=next(r for r in evaluated if r['condition']=='P' and r['structure']=='filled_box')
    assert p['absolute_occupancy_iou']==p['P_density_identity']==2/24


def test_resource_rejected_predictions_persist_nulls_and_no_partial_artifacts(tmp_path):
    records=construction_records();manifest=base.generate_predictions(records,tmp_path/'predictions',Limits(max_voxels=1))
    assert manifest['counts']=={'BASELINE_INPUT_OR_RESOURCE_REJECTED':4}
    assert all(r['prediction_voxels'] is None and r['artifacts']=={} for r in manifest['predictions'])
    assert list((tmp_path/'predictions').iterdir())==[]


def test_unsupported_condition_not_treated_as_partial_oracle(tmp_path):
    records=construction_records()[1:];records[0]['condition']='E'
    with pytest.raises(ValueError):base.generate_predictions(records,tmp_path/'predictions')


@pytest.mark.parametrize('mutation',['bytes','coordinates_with_updated_hash','unsafe_path','wrong_count'])
def test_corrupted_saved_prediction_is_rejected(tmp_path,mutation):
    manifest=base.generate_predictions(construction_records()[:1],tmp_path/'predictions');row=deepcopy(manifest['predictions'][0])
    artifact=row['artifacts']['coordinates'];path=tmp_path/'predictions'/artifact['path']
    if mutation=='bytes':
        with path.open('ab') as stream:stream.write(b'x')
    elif mutation=='coordinates_with_updated_hash':
        coordinates=np.load(path,allow_pickle=False);coordinates[0,0]+=99
        with path.open('wb') as stream:np.save(stream,coordinates,allow_pickle=False)
        artifact.update(sha256=base.sha256_file(path),bytes=path.stat().st_size)
    elif mutation=='unsafe_path':artifact['path']='../outside.npy'
    else:row['prediction_voxels']+=1
    with pytest.raises(ValueError):base._load_prediction(row,tmp_path/'predictions',Limits())


def comparison_fixture():
    _,slots,_=make_case(3)
    for slot in slots:slot['metrics']['absolute']={'occupancy_iou':max(0,slot['metrics']['aligned']['occupancy_iou']-.02)}
    for i,scene in enumerate(sorted({r['scene_id'] for r in slots})):
        for builder in ('openai','anthropic'):
            row=deepcopy(next(r for r in slots if r['scene_id']==scene and r['provider']==builder))
            row.update(condition='P',description_model=None,input_id=scene+':P',input_sha256=hashlib.sha256((scene+':P').encode()).hexdigest(),
                       attempt_id=scene+':P:'+builder,global_schedule_index=len(slots))
            slots.append(row)
    records={}
    for row in slots:
        if row['condition'] not in ('O','P'):continue
        for structure in base.STRUCTURES:
            key=(row['scene_id'],row['condition'],row['description_model'],structure)
            records[key]={**{k:row[k] for k in ('scene_id','condition','description_model','input_id','input_sha256')},
                'prediction_id':hashlib.sha256(str(key).encode()).hexdigest(),'structure':structure,'status':'success',
                'absolute_occupancy_iou':.09,'aligned_occupancy_iou':.12,'size_rank_stratum':int(row['scene_id'][-2:])}
    return list(records.values()),slots


def test_all80_comparisons_are_paired_and_equal_building_weight_with_explicit_missingness():
    predictions,slots=comparison_fixture()
    failed=next(r for r in slots if r['scene_id']=='fixture-scene-00' and r['condition']=='O' and r['provider']=='openai' and r['description_model']=='openai')
    failed.update(status='failed',metrics=None,failure_kind='validation_failure')
    unknown=next(r for r in slots if r['scene_id']=='fixture-scene-01' and r['condition']=='O' and r['provider']=='anthropic' and r['description_model']=='gemini')
    unknown.update(status='operational_unmeasurable',metrics=None,usage_known=False,completed_request_outcome=False,http_status=520)
    rejected=next(r for r in predictions if r['scene_id']=='fixture-scene-02' and r['condition']=='O' and r['description_model']=='anthropic' and r['structure']=='filled_box')
    rejected.update(status='BASELINE_INPUT_OR_RESOURCE_REJECTED',absolute_occupancy_iou=None,aligned_occupancy_iou=None)
    before=deepcopy((predictions,slots));records,pairs=base.compare_baselines(predictions,slots,fixed_design=False)
    assert (predictions,slots)==before and len(records)==80 and len(pairs)==192
    assert all(r['p_values_computed'] is False and r['planned_buildings']==3 and r['independent_buildings']<=3 for r in records)
    for builder,expected in [('openai',(7,8)),('anthropic',(7,7))]:
        for policy,count in zip(base.POLICIES,expected):
            r=next(r for r in records if (r['condition'],r['description_provider'],r['structure'],r['builder'],r['metric'],r['policy'])==('O','all','filled_box',builder,'aligned_occupancy_iou',policy))
            assert r['planned_pairs']==9 and r['common_pairs']==count
            assert r['failure0_imputed_pairs']==(1 if builder=='openai' and policy=='known_failure0' else 0)
            selected=[p for p in pairs if p['condition']=='O' and p['structure']=='filled_box' and p['builder']==builder and p['metric']=='aligned_occupancy_iou' and p['policy']==policy and p['delta'] is not None]
            by_scene=defaultdict(list)
            for p in selected:by_scene[p['scene_id']].append(p['delta'])
            assert r['mean_delta']==pytest.approx(np.mean([np.mean(v) for v in by_scene.values()]))
            assert all(p['delta']==pytest.approx(p['builder_value']-p['baseline_value']) for p in selected)
    assert all(p['delta'] is None and not p['failure0_imputed'] for p in pairs if p['attempt_id']==unknown['attempt_id'])
    assert all(p['delta'] is None for p in pairs if p['prediction_id']==rejected['prediction_id'])
    json.dumps(records,allow_nan=False)


@pytest.mark.parametrize('mutation',['duplicate','input_hash'])
def test_baseline_pair_binding_rejects_mutation(mutation):
    predictions,slots=comparison_fixture()
    if mutation=='duplicate':predictions.append(deepcopy(predictions[0]))
    else:predictions[0]['input_sha256']='0'*64
    with pytest.raises(ValueError):base.compare_baselines(predictions,slots,fixed_design=False)


def test_planned_frame_cluster_ci_matches_explicit_missing_cluster_resampling():
    values={'a':-2.,'c':5.};frame=['a','b','c','d','e']
    actual=base.building_interval(values,frame)
    rng=np.random.Generator(np.random.PCG64(20261017));draws=rng.integers(0,5,size=(2000,5))
    samples=[]
    for draw in draws:
        selected=[values[frame[i]] for i in draw if frame[i] in values]
        if selected:samples.append(float(np.mean(selected)))
    assert actual['mean_delta']==1.5 and actual['independent_buildings']==2 and actual['planned_buildings']==5
    assert actual['bootstrap_valid']==len(samples)<1900
    assert actual['status']=='UNSTABLE_BOOTSTRAP' and actual['ci95'] is None
    values['b']=1.;actual=base.building_interval(values,frame)
    samples=[np.mean([values[frame[i]] for i in draw if frame[i] in values]) for draw in draws if any(frame[i] in values for i in draw)]
    assert actual['bootstrap_valid']==len(samples)>=1900
    assert actual['ci95']==pytest.approx(np.quantile(samples,[.025,.975],method='linear'))


def test_size_distributions_keep_all48_cells_and_average_within_building():
    rows,_=comparison_fixture();result=base.size_distributions(rows)
    assert len(result)==48
    empty=[r for r in result if r['size_rank_stratum'] in (3,4)]
    assert len(empty)==16 and all(r['independent_buildings']==0 and r['q1_median_q3'] is None for r in empty)


def test_independent_geometry_resource_failure_is_missing_not_unverified_success(tmp_path,monkeypatch):
    manifest=base.generate_predictions(construction_records()[:1],tmp_path/'predictions')
    gt=Grid({(0,0,0):'stone'})
    scene=SimpleNamespace(scene_id='fixture-scene',grid=gt,bbox=bbox(1,1,1))
    monkeypatch.setattr(base,'load_scene',lambda *a,**k:scene)
    def limited(*a,**k):raise independent.IndependentGeometryResourceLimit('synthetic fixture ceiling',{'fixture':True})
    monkeypatch.setattr(independent,'check_box',limited)
    package={'descriptions':[{'scene_id':'fixture-scene','gt_descriptor':{'building_directory':'fixture-only',
        'grid_sha256':base.object_hash(gt.to_dict()),'bbox_sha256':'d'*64,'voxels_sha256':'e'*64}}],
        'cohort':{'scenes':[{'scene_id':'fixture-scene','size_rank_stratum':0}]}}
    rows=base.evaluate_predictions(manifest['predictions'],package,tmp_path/'predictions',Limits())
    assert len(rows)==2 and all(r['status']=='INDEPENDENT_GEOMETRY_RESOURCE_LIMIT' for r in rows)
    assert all(r['alignment_failure']['status']=='INDEPENDENT_GEOMETRY_RESOURCE_LIMIT' for r in rows)
    assert all(r['absolute_occupancy_iou'] is None and r['aligned_occupancy_iou'] is None and r['independent_check_status']!='PASS' for r in rows)
