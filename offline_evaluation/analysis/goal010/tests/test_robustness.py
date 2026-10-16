"""Synthetic-only fixed proxy robustness grid and independent arithmetic checks."""
from collections import Counter
from copy import deepcopy
import json
import socket

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from offline_evaluation.analysis.goal010 import factors, robustness
from offline_evaluation.analysis.goal008 import statistics as frozen
from offline_evaluation.analysis.goal010.tests.test_factors import SPEC, gate, make_case


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*a,**k):pytest.fail('network forbidden in CPU robustness fixtures')
    monkeypatch.setattr(socket.socket,'connect',denied)
    monkeypatch.setattr(socket.socket,'connect_ex',denied)


@pytest.fixture(scope='module')
def full_result():
    scores,rows,metadata=make_case();before=deepcopy((scores,rows,metadata))
    result=robustness.analyze_robustness(scores,rows,metadata,audit_gate=gate(),specification_sha256=SPEC)
    assert (scores,rows,metadata)==before
    return result,scores,rows,metadata


def record(result,feature='composite',group='proxy_stratified',scope='all',target='openai',policy='valid_only',method='pooled_spearman'):
    return next(r for r in result['records'] if (r['feature'],r['analysis_group'],r['scope_id'],r['outcome_target'],r['outcome_policy'],r['association'])==(feature,group,scope,target,policy,method))


def joined(scores,rows,metadata):
    return robustness._metadata_join(frozen.join_descriptions(scores,rows,fixed_design=False),metadata,fixed_design=False)


def test_complete1350_grid_and_n_do_not_count_builders_independently(full_result):
    result,scores,rows,metadata=full_result
    assert len(result['records'])==len({r['id'] for r in result['records']})==1350
    assert result['record_group_counts']=={'proxy_stratified':972,'proxy_within_building':108,'covariate_outcome':270}
    assert len(result['scope_counts'])==18
    assert result['source_binding']['buildings']==5 and result['source_binding']['descriptions']==15
    assert all(r['counts']['analysis_buildings']<=5 and r['p_values_computed'] is False and 'p_holm' not in r for r in result['records'])
    assert all(r['bootstrap_frame_buildings']<=5 for r in result['records'])
    assert result['goal008_frozen_results_modified'] is False
    json.dumps(result,allow_nan=False)


def test_pooled_point_and_2000_ci_match_independent_explicit_cluster_copies(full_result):
    result,scores,rows,metadata=full_result
    data=joined(scores,rows,metadata);scenes=sorted({r['scene_id'] for r in data})
    x=np.array([r['scores']['composite'] for r in data]);y=np.array([r['builders']['openai']['valid_only'] for r in data])
    clusters=np.array([scenes.index(r['scene_id']) for r in data])
    actual=record(result)
    assert actual['coefficient']==pytest.approx(spearmanr(x,y).statistic,abs=1e-13)
    rng=np.random.Generator(np.random.PCG64(20261017));draws=rng.integers(0,len(scenes),size=(2000,len(scenes)))
    coefficients=[]
    for draw in draws:
        ix=np.concatenate([np.flatnonzero(clusters==value) for value in draw])
        rx,ry=rankdata(x[ix]),rankdata(y[ix])
        if np.std(rx)>0 and np.std(ry)>0:coefficients.append(float(np.corrcoef(rx,ry)[0,1]))
    assert actual['bootstrap_valid']==len(coefficients)
    assert actual['ci95']==pytest.approx(np.quantile(coefficients,[.025,.975],method='linear'),abs=1e-12)


def test_within_center_and_rank_are_manual_building_residual_correlations(full_result):
    result,scores,rows,metadata=full_result;data=joined(scores,rows,metadata)
    for method in ('within_building_centered_pearson','within_building_midrank_association'):
        x=[];y=[]
        for scene in sorted({r['scene_id'] for r in data}):
            subset=[r for r in data if r['scene_id']==scene]
            vx=np.array([r['scores']['composite'] for r in subset]);vy=np.array([r['builders']['openai']['valid_only'] for r in subset])
            if method.endswith('midrank_association'):vx,vy=rankdata(vx),rankdata(vy)
            x.extend(vx-vx.mean());y.extend(vy-vy.mean())
        actual=record(result,group='proxy_within_building',method=method)
        assert actual['coefficient']==pytest.approx(np.corrcoef(x,y)[0,1],abs=1e-12)


@pytest.mark.parametrize('field',['input_id',*frozen.HASH_FIELDS])
def test_metadata_source_identity_mismatch_is_rejected(field):
    scores,rows,metadata=make_case(2);metadata[0][field]='changed'
    with pytest.raises(ValueError,match='identity mismatch'):joined(scores,rows,metadata)


@pytest.mark.parametrize('field,value',[('description_chars',True),('human_text_chars',-1),('gt_occupied_count',0),('gt_storage_bbox_volume',1),('gt_occupancy_density',float('nan')),('size_rank_stratum',5)])
def test_bad_covariate_metadata_fail_closed(field,value):
    scores,rows,metadata=make_case(2);metadata[0][field]=value
    with pytest.raises(ValueError):joined(scores,rows,metadata)


def test_building_gt_covariates_and_duplicate_metadata_cannot_disagree():
    scores,rows,metadata=make_case(2)
    with pytest.raises(ValueError,match='duplicate'):joined(scores,rows,metadata+[deepcopy(metadata[0])])
    metadata[0]['size_rank_stratum']=4
    with pytest.raises(ValueError,match='disagree'):joined(scores,rows,metadata)


def test_leave_one_component_out_keeps_frozen_values_and_missing_propagates():
    scores,rows,metadata=make_case(2);before=deepcopy(scores);data=joined(scores,rows,metadata)
    for output,source in zip(data,sorted(scores,key=lambda r:(r['scene_id'],r['description_provider']))):
        for key in frozen.FEATURES:assert output['scores'][key]==source['scores'][key]
        for removed in robustness.WEIGHTS:
            expected=sum(source['scores'][k]*w for k,w in robustness.WEIGHTS.items() if k!=removed)/(1-robustness.WEIGHTS[removed])
            assert output['scores']['without_'+removed]==pytest.approx(expected)
    assert scores==before
    scores[0]['scores']['dimension_accuracy']=None;scores[0]['missing_reasons']={'dimension_accuracy':'fixture missing'}
    data=joined(scores,rows,metadata);out=next(r for r in data if r['input_id']==scores[0]['input_id'])
    assert out['scores']['without_dimension_accuracy'] is not None
    assert all(out['scores']['without_'+k] is None for k in robustness.WEIGHTS if k!='dimension_accuracy')


def test_tertiles_preserve_boundary_ties_and_empty_cells():
    scores,rows,metadata=make_case(2)
    for row in metadata:row['description_chars']=100
    data=joined(scores,rows,metadata);scopes,diagnostic,bins=robustness._scopes(data)
    assert bins['description_chars']['cutpoints']==[100.,100.]
    assert bins['description_chars']['boundary_tie_counts']==[6,6]
    tertiles=[s for s in scopes if s[0].startswith('description_chars:')]
    assert [len(s[1]) for s in tertiles]==[6,0,0]
    assert bins['gt_storage_bbox_volume']['planned_values']==2
    assert bins['description_chars']['planned_values']==6


def test_failure0_only_known_completed_and_both_builder_mean_requires_both():
    scores,rows,metadata=make_case(2)
    a=next(r for r in rows if r['condition']=='O' and r['provider']=='openai')
    b=next(r for r in rows if r['condition']=='O' and r['provider']=='anthropic')
    a.update(status='failed',metrics=None,failure_kind='truncation')
    b.update(status='operational_unmeasurable',metrics=None,usage_known=False,completed_request_outcome=False,http_status=520)
    data=joined(scores,rows,metadata);first=next(r for r in data if r['input_id']==a['input_id'])
    assert first['builders']['openai']['bounded_iou_failure0']==0
    assert first['builders']['anthropic']['bounded_iou_failure0'] is None
    assert frozen._target(first,'both_builders_mean','bounded_iou_failure0') is None


def test_full_grid_preserves_failure0_and_unknown_denominators():
    scores,rows,metadata=make_case(2)
    a=next(r for r in rows if r['condition']=='O' and r['provider']=='openai')
    b=next(r for r in rows if r['condition']=='O' and r['provider']=='anthropic')
    a.update(status='failed',metrics=None,failure_kind='validation_failure')
    b.update(status='operational_unmeasurable',metrics=None,usage_known=False,completed_request_outcome=False,http_status=520)
    before=deepcopy((scores,rows,metadata))
    result=robustness.analyze_robustness(scores,rows,metadata,audit_gate=gate(),specification_sha256=SPEC)
    assert (scores,rows,metadata)==before and len(result['records'])==1350
    for target in frozen.TARGETS:
        valid=record(result,target=target); f0=record(result,target=target,policy='bounded_iou_failure0')
        assert valid['counts']['analysis_pairs']==5
        assert f0['counts']['analysis_pairs']==(6 if target=='openai' else 5)
        assert f0['counts']['failure0_converted_descriptions']==(1 if target=='openai' else 0)
        assert f0['counts']['unknown_usage_descriptions']==(0 if target=='openai' else 1)


def test_constants_and_single_building_ci_limits_remain_explicit(full_result):
    result,*_=full_result
    constant=record(result,feature='completeness',scope='description_provider:openai')
    assert constant['status']=='CONSTANT_PROXY' and constant['coefficient'] is None and constant['ci95'] is None
    small=record(result,scope='size_rank_stratum:0')
    assert small['counts']['analysis_buildings']==1 and small['ci95'] is None


def test_robustness_gate_precedes_any_join(monkeypatch):
    blocked=gate();blocked['definitions_frozen']=False
    monkeypatch.setattr(frozen,'join_descriptions',lambda *a,**k:pytest.fail('join before freeze'))
    with pytest.raises(ValueError):robustness.analyze_robustness([],[],[],audit_gate=blocked,specification_sha256=SPEC)
