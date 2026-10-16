"""Synthetic statistics checks; no historical outcomes, voxels or API access."""
from copy import deepcopy
import hashlib
import json
import socket

import numpy as np
import pytest

from offline_evaluation.analysis.goal010v2 import statistics as s

SPEC = 'a'*64


def registry():
    entries=[]
    for name in sorted(s.CORE_GEOMETRY):
        entries.append({'metric_id':name,'section':'frame','value_range':[0,1], 'direction':'higher',
                        'failure0_allowed':True,'baseline_comparable':True,'priority':'core'})
    for name,bounds,direction,baseline,zero in (
        ('volume_ratio',[0,None],'descriptive',True,False),
        ('relative_volume_error',[-1,None],'descriptive',True,False),
        ('column_count_mae',[0,None],'lower',True,False),
        ('conditional_exact_material_accuracy',[0,1],'higher',False,False),
        ('gt_material_recall',[0,1],'higher',False,True)):
        entries.append({'metric_id':name,'section':'frame','value_range':bounds,'direction':direction,
                        'failure0_allowed':zero,'baseline_comparable':baseline,'priority':'additional'})
    return {'metrics':entries}


def case(buildings=3):
    slots,bases=[],[];reg=registry();scenes=[f'fixture-{i:02d}' for i in range(buildings)]
    sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    for si,scene in enumerate(scenes):
        for condition in s.CONDITIONS:
            for di,description in enumerate(s.DESCRIPTIONS if condition in ('O','J') else (None,)):
                identity={'scene_id':scene,'condition':condition,'description_model':description,
                    'input_id':scene+condition+str(description),'input_sha256':sha(scene+condition+str(description)),
                    'gt_grid_sha256':sha(scene),'coordinate_system':'XYZ_WORLD_Y_UP_INTEGER',
                    'alignment_protocol':'server_v2.2_strict_translation'}
                values={e['metric_id']:.2+.05*si+.02*di for e in reg['metrics']}
                for builder in s.BUILDERS:
                    slots.append({**identity,'provider':builder,'global_schedule_index':len(slots),
                        'status':'success','failure0_eligible':False,'usage_known':True,
                        'completed_request_outcome':True,'failure_kind':None,'http_status_code':200,
                        'metrics':{frame:dict(values) for frame in s.FRAMES},
                        'null_reasons':{frame:{} for frame in s.FRAMES},
                        'provenance':{'fixture':True,'stored_alignment_shift_xyz':[1,0,0]}})
                if condition in ('O','P'):
                    for structure in s.STRUCTURES:
                        bases.append({**identity,'prediction_id':identity['input_id']+structure,'structure':structure,
                            'status':'success','metrics':{frame:{k:v+.1 for k,v in values.items()} for frame in s.FRAMES},
                            'null_reasons':{frame:{} for frame in s.FRAMES},
                            'provenance':{'fixture':True,'stored_alignment_shift_xyz':[-2,0,0]}})
    gate={'status':'PASS','input_kind':'fixture','phase0_binding':'PASS','definitions_frozen':True,
          'specification_sha256':SPEC,'planned_scene_ids':scenes}
    return slots,bases,gate,reg


def run(case):
    rows,bases,gate,reg=case
    return s.summarize(rows,bases,gate=gate,specification_sha256=SPEC,registry=reg)


def null_row(row,reason='known validation failure',status='failed'):
    row.update(status=status, failure_kind='validation_failure',failure0_eligible=status=='failed')
    for frame in s.FRAMES:
        row['metrics'][frame]={k:None for k in row['metrics'][frame]}
        row['null_reasons'][frame]={k:reason for k in row['metrics'][frame]}


def target(result,name='occupancy_iou',policy='common_valid',scope='all'):
    return next(r for r in result['paired_summary'] if r['builder']=='openai' and r['condition']=='O'
        and r['structure']=='filled_box' and r['frame']=='aligned' and r['metric_id']==name
        and r['policy']==policy and r['description_provider']==scope)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*a,**k):pytest.fail('CPU statistics fixtures must never open network connections')
    monkeypatch.setattr(socket.socket,'connect',deny);monkeypatch.setattr(socket.socket,'connect_ex',deny)


@pytest.fixture(scope='module')
def completed():
    data=case();before=deepcopy(data);result=run(data)
    assert data==before
    return result


def test_complete_grid_reliability_denominators_and_no_artificial_material(completed):
    r=completed
    # Geometry: six bounded core metrics x2 policies + three unbounded x1.
    assert len(r['paired_summary'])==40*(6*2+3)==600
    assert len(r['paired_rows'])==24*2*2*(6*2+3)==1440
    assert len(r['metric_denominators'])==20*2*(7*2+4)==720
    assert len(r['reliability'])==8 and sum(x['planned_slots'] for x in r['reliability'])==48
    assert all('material' not in row['metric_id'] for row in r['paired_summary'])
    assert all(row['independent_buildings']==3 and row['p_values_computed'] is False for row in r['paired_summary'])
    assert target(r)['mean_delta']==pytest.approx(-.1)
    assert target(r)['ci95']==pytest.approx([-.1,-.1])
    assert r['metadata']['research_api_requests_sent']==0
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('key,value',[('status','STOP'),('phase0_binding','PENDING'),('definitions_frozen',False),
    ('specification_sha256','b'*64),('planned_scene_ids',['b','a']),('input_kind','unverified')])
def test_gate_precedes_data_access(key,value,monkeypatch):
    rows,bases,gate,reg=case();gate[key]=value
    monkeypatch.setattr(s,'validate_inputs',lambda *a,**k:pytest.fail('rows touched before gate'))
    with pytest.raises(ValueError):s.summarize(rows,bases,gate=gate,specification_sha256=SPEC,registry=reg)


def test_common_valid_and_known_failure0_are_separate_and_unbounded_not_imputed():
    data=case();rows=data[0];row=rows[0];null_row(row)
    result=run(data)
    assert target(result)['eligible_pairs']==8
    assert target(result,policy='known_failure0')['eligible_pairs']==9
    assert target(result,policy='known_failure0')['failure0_imputed_pairs']==1
    assert target(result,'volume_ratio')['eligible_pairs']==8
    assert not any(r['metric_id']=='volume_ratio' and r['policy']=='known_failure0' for r in result['paired_summary'])
    assert not any(r['metric_id']=='conditional_exact_material_accuracy' and r['policy']=='known_failure0' for r in result['metric_denominators'])
    assert result['reliability'][0]['identified_output_failures']==1


def test_unknown942_style_missing_never_imputed_even_with_failure_label():
    data=case();row=data[0][0];null_row(row,'operational missing','operational_unmeasurable')
    row.update(failure_kind='truncation',usage_known=False,completed_request_outcome=False,http_status_code=520)
    result=run(data)
    assert target(result,policy='known_failure0')['eligible_pairs']==8
    assert target(result,policy='known_failure0')['failure0_imputed_pairs']==0
    assert result['reliability'][0]['operational_unknown']==1


def test_successful_empty_precision_and_resource_null_stay_null_in_failure0():
    data=case();row=data[0][0]
    for frame in s.FRAMES:
        row['metrics'][frame]['occupancy_precision']=None
        row['null_reasons'][frame]['occupancy_precision']='EMPTY_PREDICTION_PRECISION_UNDEFINED'
        row['metrics'][frame]['surface_f1_at_1']=None
        row['null_reasons'][frame]['surface_f1_at_1']='RESOURCE_LIMIT'
    result=run(data)
    for name in ('occupancy_precision','surface_f1_at_1'):
        v=target(result,name,'known_failure0')
        assert v['eligible_pairs']==8 and v['failure0_imputed_pairs']==0
        assert sum(v['builder_null_reason_counts'].values())==1


def test_baseline_missing_is_never_zero_and_both_reasons_preserved():
    data=case();null_row(data[0][0]);base=data[1][0]
    null_row(base,'CPU resource limit','BASELINE_INPUT_OR_RESOURCE_REJECTED')
    result=run(data)
    assert target(result)['exclusion_counts']['both_unmeasured']==1
    assert target(result,policy='known_failure0')['eligible_pairs']==8
    assert target(result,policy='known_failure0')['failure0_imputed_pairs']==0
    assert any(r['baseline_null_reason']=='CPU resource limit' for r in result['exclusions'])


def test_equal_building_mean_avoids_counting_more_provider_pairs_as_more_buildings():
    data=case(2);rows,bases,gate,reg=data
    for row in rows:
        for frame in s.FRAMES:row['metrics'][frame]['occupancy_iou']=.9 if row['scene_id']=='fixture-00' else .1
    for row in bases:
        for frame in s.FRAMES:row['metrics'][frame]['occupancy_iou']=0.
    for row in rows:
        if row['scene_id']=='fixture-00' and row['condition']=='O' and row['provider']=='openai' and row['description_model']!='openai':
            null_row(row,'missing unknown','operational_unmeasurable');row.update(usage_known=False,completed_request_outcome=False,http_status_code=520)
    result=run(data);v=target(result)
    assert v['eligible_pairs']==4 and v['independent_buildings']==2
    assert v['mean_delta']==pytest.approx(.5) # not (.9+.1+.1+.1)/4


def test_bootstrap_matches_explicit_planned_cluster_draws_with_missing_building():
    scenes=['a','b','c','d'];values={'a':-2.,'d':7.}
    result=s.building_interval(values,scenes)
    rng=np.random.Generator(np.random.PCG64(20261017));draws=rng.integers(0,4,size=(2000,4))
    samples=[]
    for draw in draws:
        x=[values[scenes[i]] for i in draw if scenes[i] in values]
        if x:samples.append(np.mean(x))
    assert result['mean_delta']==2.5
    assert result['bootstrap_valid']==len(samples)
    if len(samples)>=1900:assert result['ci95']==pytest.approx(np.quantile(samples,[.025,.975],method='linear'))
    else:assert result['status']=='ESTIMATE_WITHOUT_STABLE_CI' and result['ci95'] is None


def test_missing_buildings_retained_in_bootstrap_can_prevent_ci():
    scenes=[f's{i:02d}' for i in range(60)]
    r=s.building_interval({scenes[0]:0.,scenes[1]:1.},scenes)
    assert r['planned_buildings']==60 and r['independent_buildings']==2
    assert r['bootstrap_valid']<1900 and r['ci95'] is None
    assert r['status']=='ESTIMATE_WITHOUT_STABLE_CI'


def test_one_building_point_retained_without_ci():
    r=s.building_interval({'a':2.},['a','b'])
    assert r['mean_delta']==2 and r['ci95'] is None and r['bootstrap_attempted']==0


def test_rank_reversal_uses_exact_pair_intersection_and_direction():
    data=case()
    for row in data[0]:
        for frame in s.FRAMES:
            row['metrics'][frame]['surface_f1_at_1']=.9
            row['metrics'][frame]['column_count_mae']=.05
    row=data[0][0]
    for frame in s.FRAMES:
        row['metrics'][frame]['surface_f1_at_1']=None;row['null_reasons'][frame]['surface_f1_at_1']='RESOURCE_LIMIT'
    result=run(data)
    def find(name):return next(r for r in result['rank_reversals'] if r['builder']=='openai' and r['condition']=='O' and r['description_provider']=='all' and r['structure']=='filled_box' and r['frame']=='aligned' and r['metric_id']==name)
    r=find('surface_f1_at_1')
    assert r['common_pairs']==8 and r['same_pair_set_for_both_metrics']
    assert r['directional_reversal'] is True and r['reference_iou_delta']==pytest.approx(-.1)
    assert find('column_count_mae')['directional_reversal'] is True
    assert find('volume_ratio')['directional_reversal'] is None


@pytest.mark.parametrize('mutation',['duplicate_slot','missing_slot','duplicate_baseline','missing_baseline','input','GT','coordinate','protocol','unknown_f0','successful_f0','missing_reason','NaN','bool','outside_range','nonsuccess_value','HTTP_alias'])
def test_source_design_and_metric_corruption_fails_closed(mutation):
    data=case();rows,bases,gate,reg=data
    if mutation=='duplicate_slot':rows.append(deepcopy(rows[0]))
    elif mutation=='missing_slot':rows.pop()
    elif mutation=='duplicate_baseline':bases.append(deepcopy(bases[0]))
    elif mutation=='missing_baseline':bases.pop()
    elif mutation=='input':bases[0]['input_sha256']='f'*64
    elif mutation=='GT':bases[0]['gt_grid_sha256']='f'*64
    elif mutation=='coordinate':bases[0]['coordinate_system']='YXZ'
    elif mutation=='protocol':bases[0]['alignment_protocol']='new metric optimized shift'
    elif mutation=='unknown_f0':null_row(rows[0]);rows[0].update(usage_known=False,completed_request_outcome=False,http_status_code=520)
    elif mutation=='successful_f0':rows[0]['failure0_eligible']=True
    elif mutation=='missing_reason':rows[0]['metrics']['absolute']['occupancy_iou']=None
    elif mutation=='NaN':rows[0]['metrics']['absolute']['occupancy_iou']=float('nan')
    elif mutation=='bool':rows[0]['metrics']['absolute']['occupancy_iou']=True
    elif mutation=='outside_range':rows[0]['metrics']['absolute']['occupancy_iou']=2.
    elif mutation=='nonsuccess_value':rows[0].update(status='failed',failure_kind='validation_failure',failure0_eligible=True)
    else:rows[0]['http_status']=520
    with pytest.raises((ValueError,AssertionError)):run(data)


@pytest.mark.parametrize('mutation',['material_baseline','unbounded_zero','lower_zero','conditional_zero'])
def test_registry_cannot_enable_artificial_material_or_invalid_failure0(mutation):
    data=case();reg=data[3]
    if mutation=='material_baseline':entry=next(r for r in reg['metrics'] if r['metric_id']=='gt_material_recall');entry['baseline_comparable']=True
    elif mutation=='unbounded_zero':entry=next(r for r in reg['metrics'] if r['metric_id']=='volume_ratio');entry['failure0_allowed']=True
    elif mutation=='lower_zero':reg['metrics'][0]['direction']='lower'
    else:entry=next(r for r in reg['metrics'] if r['metric_id']=='conditional_exact_material_accuracy');entry['failure0_allowed']=True
    with pytest.raises(ValueError):run(data)


def test_different_per_prediction_shifts_allowed_under_same_protocol(completed):
    pair=completed['paired_rows'][0]
    assert pair['builder_provenance']['stored_alignment_shift_xyz']!=pair['baseline_provenance']['stored_alignment_shift_xyz']
    assert pair['alignment_protocol']=='server_v2.2_strict_translation'


def test_actual_registry_contract_and_full_grid_on_synthetic_rows():
    from offline_evaluation.analysis.goal010v2 import metrics
    rows,bases,gate,_=case(2);reg=metrics.metric_registry()
    entries=[r for r in reg['metrics'] if r['section']=='frame']
    for row in rows+bases:
        for frame in s.FRAMES:
            row['metrics'][frame]={e['metric_id']:(1 if e['unit']=='count' else .5) for e in entries}
            row['null_reasons'][frame]={}
    for row in bases:
        for frame in s.FRAMES:
            for name in metrics.MATERIAL_METRIC_IDS:
                row['metrics'][frame][name]=None
                row['null_reasons'][frame][name]='ARTIFICIAL_BASELINE_MATERIAL_NOT_INTERPRETED'
    result=s.summarize(rows,bases,gate=gate,specification_sha256=SPEC,registry=reg)
    assert len(result['paired_summary'])==2320
    assert len(result['metric_denominators'])==2640
    assert len(result['rank_reversals'])==1560
    assert len(result['paired_rows'])==111360//30
    assert all(r['ordinal_preference'] in ('NUMERIC_TIE','NO_FAVORABLE_ORDER') for r in result['paired_summary'])
    assert all(r['directional_reversal'] is None for r in result['rank_reversals'])
