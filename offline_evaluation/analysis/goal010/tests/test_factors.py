"""Synthetic paired-factor tests: no research source reads or API calls."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import socket

import numpy as np
import pytest

from offline_evaluation.analysis.goal010 import factors
from offline_evaluation.analysis.goal008 import statistics as frozen

SPEC = 'a'*64


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail('network is forbidden in Goal010 CPU fixtures')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket.socket, 'connect_ex', denied)


def gate():
    return {'status': 'PASS', 'input_kind': 'fixture', 'phase0_binding': 'PASS', 'definitions_frozen': True,
            'specification_sha256': SPEC}


def make_case(buildings=5):
    rows, proxies, metadata = [], [], []
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    for building in range(buildings):
        scene = f'fixture-scene-{building:02d}'
        for d, description in enumerate(factors.PROVIDERS):
            input_id = scene+':'+description+':O'
            scores = {'strict_material_f1': ((building*2+d+1)%11)/11,
                      'coarse_material_f1': ((building+d+3)%9)/9,
                      'dimension_accuracy': ((building*3+d+2)%13)/13, 'completeness': 1.0 if d<2 else .8}
            scores['composite'] = sum(scores[k]*w for k,w in {'strict_material_f1':.30,'coarse_material_f1':.25,'dimension_accuracy':.25,'completeness':.20}.items())
            identity = {'scene_id':scene,'description_provider':description,'input_id':input_id,
                        **{k:sha((scene if k.startswith('gt_') else input_id)+k) for k in frozen.HASH_FIELDS}}
            proxies.append({**identity, 'scores':scores})
            occupied=10+building*4; volume=100+building*11
            metadata.append({**identity,'description_chars':100+building*17+d*10,'human_text_chars':50+building*7+d*4,
                             'gt_occupied_count':occupied,'gt_storage_bbox_volume':volume,'gt_occupancy_density':occupied/volume,
                             'size_rank_stratum':building%5})
            for c,condition in enumerate(factors.CONDITIONS):
                for b,builder in enumerate(factors.BUILDERS):
                    row={**{k:v for k,v in identity.items() if k!='description_provider'},'description_model':description,
                         'provider':builder,'condition':condition,'global_schedule_index':len(rows),
                         'attempt_id':input_id+':'+condition+':'+builder,'status':'success','attempted':True,
                         'usage_known':True,'completed_request_outcome':True,'failure_kind':None,'http_status':200,
                         'source_plan_identity':sha('fixture-plan-'+builder), 'source_entry_hash':sha('entry-'+str(len(rows))),
                         'metrics':{'metric_version':'server_v2.2_strict_translation','aligned':{
                             'occupancy_iou':.08+.03*building+.04*d+.01*b+.02*c,
                             'material_aware_iou':.02+.01*building+.02*d+.005*b+.01*c,
                             'dimension_mae':3+.5*building-d+.2*b+c}}}
                    if condition=='J':
                        row['input_id']=scene+':'+description+':J';row['input_sha256']=sha(row['input_id'])
                    rows.append(row)
    return proxies,rows,metadata


def analyze(rows):
    return factors.analyze_factors(rows,audit_gate=gate(),specification_sha256=SPEC)


@pytest.mark.parametrize('field,value',[('status','STOP'),('input_kind','unverified'),('phase0_binding','PENDING'),
                                      ('definitions_frozen',False),('specification_sha256','b'*64)])
def test_gate_precedes_row_access(field,value,monkeypatch):
    blocked=gate();blocked[field]=value
    monkeypatch.setattr(factors.original,'validate_rows',lambda *a,**k:pytest.fail('row inspection before gate'))
    with pytest.raises(ValueError): factors.analyze_factors([],audit_gate=blocked,specification_sha256=SPEC)


def test_all36_paired_cells_and_direction_without_pvalues():
    _,rows,_=make_case(); before=deepcopy(rows); result=analyze(rows)
    assert rows==before
    assert len(result['records'])==36 and len(result['pairs'])==180 and result['exclusions']==[]
    assert len({r['id'] for r in result['records']})==36
    for row in result['records']:
        step=factors.PROVIDERS.index(row['description_provider_b'])-factors.PROVIDERS.index(row['description_provider_a'])
        expected=step*{'aligned_occupancy_iou':.04,'aligned_material_aware_iou':.02,'aligned_dimension_mae':-1}[row['metric']]
        assert row['mean_delta']==pytest.approx(expected)
        assert row['ci95']==pytest.approx([expected,expected])
        assert row['independent_buildings']==row['common_valid_pairs']==5
        assert row['p_values_computed'] is False and 'p_holm' not in row
    assert all(p['a_source']['scene_id']==p['b_source']['scene_id'] for p in result['pairs'])
    json.dumps(result,allow_nan=False)


def test_common_valid_excludes_known_failures_and_unknowns_without_zero():
    _,rows,_=make_case()
    failed=next(r for r in rows if r['scene_id']=='fixture-scene-00' and r['condition']=='O' and r['provider']=='openai' and r['description_model']=='openai')
    failed.update(status='failed',metrics=None,failure_kind='validation_failure')
    unknown=next(r for r in rows if r['scene_id']=='fixture-scene-01' and r['condition']=='J' and r['provider']=='openai' and r['description_model']=='gemini')
    unknown.update(status='operational_unmeasurable',metrics=None,usage_known=False,completed_request_outcome=False,http_status=520)
    result=analyze(rows)
    assert len(result['exclusions'])==12
    assert Counter(r['independent_buildings'] for r in result['records'])=={5:24,4:12}
    assert {r['a_status'] for r in result['exclusions']}|{r['b_status'] for r in result['exclusions']}=={'success','failed','operational_unmeasurable'}
    assert not any(p['a_source']['attempt_id']==failed['attempt_id'] or p['b_source']['attempt_id']==unknown['attempt_id'] for p in result['pairs'])


@pytest.mark.parametrize('mutation',['duplicate','omitted','metric_version','metric_nan','alias_disagreement','failure_metrics'])
def test_invalid_design_and_outcomes_rejected(mutation):
    _,rows,_=make_case(2)
    if mutation=='duplicate': rows.append(deepcopy(rows[0]))
    elif mutation=='omitted': rows.pop()
    elif mutation=='metric_version': rows[0]['metrics']['metric_version']='old'
    elif mutation=='metric_nan': rows[0]['metrics']['aligned']['occupancy_iou']=float('nan')
    elif mutation=='alias_disagreement': rows[0]['http_status_code']=520
    else: rows[0].update(status='failed',failure_kind='truncation')
    with pytest.raises(ValueError): analyze(rows)


def test_mean_cluster_bootstrap_matches_explicit_pcg64_draws():
    values=np.array([-2.,1.,4.,8.])
    rng=np.random.Generator(np.random.PCG64(20261017))
    samples=rng.integers(0,4,size=(2000,4))
    expected=np.quantile(values[samples].mean(axis=1),[.025,.975],method='linear')
    result=factors.mean_interval(values)
    assert result['mean_delta']==pytest.approx(values.mean())
    assert result['ci95']==pytest.approx(expected)
    assert result['bootstrap_valid']==2000


def test_one_building_keeps_point_but_no_ci():
    _,rows,_=make_case(1);result=analyze(rows)
    assert all(r['independent_buildings']==1 and r['mean_delta'] is not None and r['ci95'] is None for r in result['records'])
