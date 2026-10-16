"""Synthetic proxy associations only: no real scores, sources, API or keys."""
from copy import deepcopy
import hashlib
import json
import socket

import numpy as np
import pytest
from scipy.stats import pearsonr, rankdata, spearmanr

from offline_evaluation.analysis.goal008 import statistics as stats


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('network forbidden in Goal008 statistics fixtures')
    monkeypatch.setattr(socket.socket, 'connect', denied)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(buildings=4):
    scores, slots = [], []
    for building in range(buildings):
        for description_index, description in enumerate(stats.PROVIDERS):
            scene = f'fixture-building-{building:02d}'
            identity = f'{scene}:{description}:O'
            x = (3*building + description_index + 1) / (3*buildings + 1)
            source = {'scene_id': scene, 'description_provider': description, 'input_id': identity,
                      **{name: digest(identity + name) for name in stats.HASH_FIELDS},
                      'scores': {name: (x + i*.03) % 1 for i, name in enumerate(stats.FEATURES)}}
            scores.append(source)
            for provider in stats.BUILDERS:
                slots.append({**{k: source[k] for k in ('scene_id','input_id',*stats.HASH_FIELDS)},
                    'description_model': description, 'provider': provider, 'condition': 'O',
                    'global_schedule_index': len(slots), 'attempt_id': identity + ':' + provider,
                    'status': 'success', 'failure_kind': None, 'usage_known': True, 'completed_request_outcome': True,
                    'metrics': {'metric_version': 'server_v2.2_strict_translation',
                                'aligned': {'occupancy_iou': .1+.7*x if provider=='openai' else .8-.5*x}}})
    return scores, slots


def gate():
    return {'status': 'PASS', 'input_kind': 'fixture', 'phase0_binding': 'PASS', 'definitions_frozen': True,
            'definition_sha256': stats.definition_sha256()}


def analyze(scores, slots):
    return stats.analyze_correlations(scores, slots, audit_gate=gate(), definition_sha256=stats.definition_sha256())


def record(result, *, feature='strict_material_f1', target='openai', policy='valid_only',
           association='pooled_spearman', provider=None):
    return next(r for r in result['associations'] if (r['feature'],r['outcome_target'],r['outcome_policy'],r['association'],r['description_provider'])
                == (feature,target,policy,association,provider))


@pytest.mark.parametrize('field,value', [('status','PENDING'),('phase0_binding','PENDING'),('definitions_frozen',False),
                                       ('definition_sha256','0'*64),('input_kind','unknown')])
def test_gate_and_definition_freeze_precede_join(field,value,monkeypatch):
    current = gate(); current[field] = value
    monkeypatch.setattr(stats, 'join_descriptions', lambda *_a, **_k: pytest.fail('join before gate'))
    with pytest.raises(ValueError):
        stats.analyze_correlations([], [], audit_gate=current, definition_sha256=stats.definition_sha256())


@pytest.mark.parametrize('field', ['input_id',*stats.HASH_FIELDS])
def test_source_bound_join_rejects_every_identity_mismatch(field):
    scores,slots=fixture(2); slots[0][field]='changed' if field=='input_id' else '0'*64
    with pytest.raises(ValueError,match='source binding mismatch'):
        stats.join_descriptions(scores,slots,fixed_design=False)


def test_missing_hash_alias_is_never_guessed():
    scores,slots=fixture(2);slots[0]['description_source_sha']=slots[0].pop('source_sha256')
    with pytest.raises(ValueError,match='no hash aliases'):
        stats.join_descriptions(scores,slots,fixed_design=False)


def test_duplicate_description_duplicate_builder_or_missing_builder_rejected():
    scores,slots=fixture(2)
    with pytest.raises(ValueError,match='unique'):
        stats.join_descriptions(scores+[deepcopy(scores[0])],slots,fixed_design=False)
    with pytest.raises(ValueError,match='duplicated'):
        stats.join_descriptions(scores,slots+[deepcopy(slots[0])],fixed_design=False)
    with pytest.raises(ValueError,match='exactly both'):
        stats.join_descriptions(scores,slots[:-1],fixed_design=False)


def test_all360_records_preserve_original_input_and_builder_repetition_counts():
    scores,slots=fixture();before=deepcopy((scores,slots))
    result=analyze(scores,slots)
    assert (scores,slots)==before
    assert len(result['associations'])==len({r['id'] for r in result['associations']})==360
    assert result['source_binding']['matched_descriptions']==12
    assert result['source_binding']['builder_observations']==24 and result['source_binding']['buildings']==4
    assert result['conventional_rq2_status']=='NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES'
    assert result['goal007_confirmatory_tests_reexecuted'] is False
    assert all(r['p_values_computed'] is False and 'p_holm' not in r for r in result['associations'])
    pooled=record(result)
    assert pooled['counts']['analysis_pairs']==12 and pooled['counts']['analysis_buildings']==4
    assert pooled['coefficient']==pytest.approx(1)
    assert record(result,target='anthropic')['coefficient']==pytest.approx(-1)
    assert record(result,target='both_builders_mean')['counts']['analysis_pairs']==12
    degenerate=[r for r in result['associations'] if r['status']=='NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING']
    assert len(degenerate)==180 and all(r['coefficient'] is None and r['ci95'] is None for r in degenerate)
    assert all(r['counts']['complete_pairs']==4 and r['counts']['analysis_pairs']==0 for r in degenerate)
    json.dumps(result,allow_nan=False)


def test_tied_spearman_matches_scipy_exactly():
    x=np.array([.1,.1,.3,.5,.5,.8]); y=np.array([.3,.8,.3,.2,.2,.9])
    assert np.array_equal(stats._midranks(x),rankdata(x,method='average'))
    actual=stats._weighted_correlations(stats._midranks(x),stats._midranks(y),np.ones((1,len(x))))[0]
    assert actual==pytest.approx(spearmanr(x,y).statistic,abs=1e-14)
    ties=stats._ties(x)
    assert ties=={'unique_values':4,'tie_groups':2,'tied_observations':4,'constant':False}


def test_bootstrap_weighted_ranks_match_explicit_duplicate_cluster_copies():
    x=np.array([.1,.1,.5,.2,.8,.8]);y=np.array([.9,.4,.5,.5,.2,.3]);clusters=np.array([0,0,0,1,1,1])
    counts=np.array([[2,0],[0,2],[1,1],[3,1]],dtype=np.int64)
    weights=counts[:,clusters]
    rx,ry=stats._weighted_midranks(x,weights),stats._weighted_midranks(y,weights)
    actual=stats._weighted_correlations(rx,ry,weights)
    for i,w in enumerate(weights):
        expanded_x,expanded_y=np.repeat(x,w),np.repeat(y,w)
        assert actual[i]==pytest.approx(spearmanr(expanded_x,expanded_y).statistic,abs=1e-14)
        for j in range(len(x)):
            if w[j]:assert rx[i,j]==rankdata(expanded_x)[np.cumsum(w)[j]-1]


@pytest.mark.parametrize('method', ['within_building_centered_pearson','within_building_midrank_association'])
def test_within_bootstrap_duplicate_clusters_stay_separate_and_not_globally_reranked(method):
    scores,slots=fixture(2)
    for i,score in enumerate(scores):score['scores']['strict_material_f1']=[.1,.2,.8,.7,.7,.9][i]
    for i,slot in enumerate(slots):
        if slot['provider']=='openai':slot['metrics']['aligned']['occupancy_iou']=[.8,.3,.7,.1,.1,.9][i//2]
    joined=stats.join_descriptions(scores,slots,fixed_design=False)
    x,y,clusters,counts=stats._vectors(joined,'strict_material_f1','openai','valid_only',method,{f'fixture-building-{i:02d}':i for i in range(2)})
    weights=np.array([[2,1],[1,2]])[:,clusters]
    actual=stats._weighted_correlations(x,y,weights)
    for i,w in enumerate(weights):
        assert actual[i]==pytest.approx(pearsonr(np.repeat(x,w),np.repeat(y,w)).statistic,abs=1e-14)
    assert counts['analysis_buildings']==2 and counts['analysis_pairs']==6
    if method=='within_building_midrank_association':
        assert not np.isclose(actual[0],spearmanr(np.repeat(x,weights[0]),np.repeat(y,weights[0])).statistic)


@pytest.mark.parametrize('failure',sorted(stats.DEFINITIVE_FAILURES))
def test_known_failure0_is_separate_and_two_builder_mean_requires_both(failure):
    scores,slots=fixture(3);slots[0].update(status='failed',metrics=None,failure_kind=failure)
    result=analyze(scores,slots)
    assert record(result)['counts']['complete_pairs']==8
    assert record(result,policy='bounded_iou_failure0')['counts']['complete_pairs']==9
    assert record(result,target='both_builders_mean')['counts']['complete_pairs']==8
    assert record(result,target='both_builders_mean',policy='bounded_iou_failure0')['counts']['complete_pairs']==9
    outcome=next(r for r in result['joined_descriptions'] if r['input_id']==scores[0]['input_id'])['builders']['openai']
    assert outcome['valid_only'] is None and outcome['bounded_iou_failure0']==0 and outcome['failure0_applied']
    assert record(result,policy='bounded_iou_failure0')['counts']['failure0_converted_descriptions']==1
    assert record(result)['counts']['failure0_converted_descriptions']==0


def test_operational_unknown_and_other_failure_are_never_failure0():
    scores,slots=fixture(3)
    slots[0].update(status='operational_unmeasurable',metrics=None,failure_kind='http_failure',usage_known=False,completed_request_outcome=False,http_status_code=520)
    slots[2].update(status='failed',metrics=None,failure_kind='refusal')
    result=analyze(scores,slots)
    assert record(result)['counts']['complete_pairs']==7
    assert record(result,policy='bounded_iou_failure0')['counts']['complete_pairs']==7
    assert record(result,target='both_builders_mean',policy='bounded_iou_failure0')['counts']['complete_pairs']==7
    assert all(not r['builders']['openai']['failure0_applied'] for r in result['joined_descriptions'])
    slots[0].update(usage_known=True,completed_request_outcome=True,status='failed',failure_kind='truncation')
    with pytest.raises(ValueError,match='operational unknown'):
        analyze(scores,slots)


def test_feature_missingness_is_feature_specific_with_explicit_reason():
    scores,slots=fixture(3);scores[0]['scores']['dimension_accuracy']=None
    with pytest.raises(ValueError,match='missing reason'):
        analyze(scores,slots)
    scores[0]['missing_reasons']={'dimension_accuracy':'fixture_unavailable'}
    result=analyze(scores,slots)
    assert record(result)['counts']['complete_pairs']==9
    assert record(result,feature='dimension_accuracy')['counts']['complete_pairs']==8
    assert record(result,feature='dimension_accuracy')['counts']['missing_proxy']==1
    assert record(result,feature='dimension_accuracy',association='within_building_centered_pearson')['counts']['analysis_pairs']==8


def test_constants_and_singleton_buildings_have_explicit_diagnostics():
    scores,slots=fixture(3)
    for score in scores:score['scores']['completeness']=1
    for slot in slots:slot['metrics']['aligned']['occupancy_iou']=.5
    result=analyze(scores,slots)
    assert record(result)['status']=='CONSTANT_OUTCOME'
    assert record(result,feature='completeness')['status']=='CONSTANT_BOTH'
    assert record(result,feature='completeness',association='within_building_midrank_association')['counts']['constant_proxy_buildings']==3
    assert all(r['coefficient'] is None and r['bootstrap_attempted']==0 for r in result['associations'])


def test_unstable_bootstrap_keeps_point_but_nulls_interval():
    scores,slots=fixture(2)
    selected=[s for s in scores if s['description_provider']=='openai']
    joined=stats.join_descriptions(scores,slots,fixed_design=False)
    selected=[r for r in joined if r['description_provider']=='openai']
    record=stats._association(selected,'strict_material_f1','openai','valid_only','pooled_spearman','openai',
                              {'fixture-building-00':0,'fixture-building-01':1},stats._draw_counts(2))
    assert record['coefficient']==pytest.approx(1)
    assert record['status']=='ESTIMATE_WITHOUT_CLUSTER_CI' and record['ci_status']=='UNSTABLE_BOOTSTRAP'
    assert record['ci95'] is None and 850<record['bootstrap_valid']<1150
    assert record['bootstrap_valid']+record['bootstrap_invalid']==2000


def test_deterministic_cluster_draws_and_identical_reanalysis():
    counts=stats._draw_counts(4)
    assert np.array_equal(counts,stats._draw_counts(4))
    assert counts.shape==(2000,4) and np.all(counts.sum(axis=1)==4)
    assert np.any(counts>1)
    scores,slots=fixture(2)
    assert analyze(scores,slots)==analyze(scores,slots)


def test_pooled_cluster_percentile_interval_matches_expanded_scipy_reference():
    scores,slots=fixture(3)
    x=[.1,.1,.3,.5,.5,.9,.8,.7,.3]
    y=[.6,.2,.7,.3,.3,.8,.4,.9,.1]
    for i,score in enumerate(scores):score['scores']['strict_material_f1']=x[i]
    for i,slot in enumerate(slots):
        if slot['provider']=='openai':slot['metrics']['aligned']['occupancy_iou']=y[i//2]
    result=analyze(scores,slots); actual=record(result)
    clusters=np.repeat(np.arange(3),3)
    reference=[]
    for weights in stats._draw_counts(3)[:,clusters]:
        reference.append(spearmanr(np.repeat(x,weights),np.repeat(y,weights)).statistic)
    assert actual['coefficient']==pytest.approx(spearmanr(x,y).statistic,abs=1e-14)
    assert actual['bootstrap_valid']==2000 and actual['bootstrap_invalid']==0
    assert actual['ci95']==pytest.approx(np.quantile(reference,[.025,.975],method='linear'),abs=1e-14)


def test_within_singletons_are_excluded_without_imputation_and_empty_strata_retained():
    scores,slots=fixture(2)
    for score in scores[:2]:
        score['scores']['strict_material_f1']=None
        score['missing_reasons']={'strict_material_f1':'fixture_missing'}
    result=analyze(scores,slots)
    within=record(result,association='within_building_centered_pearson')
    assert within['counts']['complete_pairs']==4 and within['counts']['analysis_pairs']==3
    assert within['counts']['excluded_singleton_buildings']==1 and within['counts']['analysis_buildings']==1
    assert within['ci_status']=='INSUFFICIENT_BUILDINGS' and within['coefficient'] is not None
    selected_scores=[score for score in scores if score['description_provider']=='openai']
    selected_slots=[slot for slot in slots if slot['description_model']=='openai']
    reduced=analyze(selected_scores,selected_slots)
    assert len(reduced['associations'])==360
    assert record(reduced,provider='gemini')['status']=='INSUFFICIENT_PAIRS'


def test_real_input_kind_requires_actual180_by60_inventory_before_analysis():
    scores,slots=fixture(2);actual=gate();actual.update(input_kind='verified_sources',matched_description_count=180)
    with pytest.raises(ValueError,match='60 buildings by3'):
        stats.analyze_correlations(scores,slots,audit_gate=actual,definition_sha256=stats.definition_sha256())


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf'),-.1,1.1,'0.5'])
def test_proxy_nonfinite_wrong_type_or_range_never_enters_statistics(bad):
    scores,slots=fixture(2);scores[0]['scores']['composite']=bad
    with pytest.raises(ValueError,match='finite number'):
        stats.join_descriptions(scores,slots,fixed_design=False)


def test_fixed_definitions_explicitly_preserve_exploratory_and_original_rq2_scope():
    text=stats.DEFINITION.read_text()
    assert '20261016' in text and '2000' in text and '1900' in text
    assert '360 records' in text and 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES' in text
    assert 'not a retrospective' in text and 'not free-form prose' in text
    assert 'not globally reranked' in text and 'not an\nequal-building mean' in text
