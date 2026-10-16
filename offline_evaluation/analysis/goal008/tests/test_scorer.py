from decimal import Decimal
import copy
import pytest
from offline_evaluation.analysis.goal008 import scorer
from offline_evaluation.server_v2.core import Grid


@pytest.fixture
def inputs():
    grid=Grid({(0,0,0):'stonebrick',(1,0,0):'glass'})
    bbox={'xmin':0,'xmax':1,'ymin':0,'ymax':0,'zmin':0,'zmax':0}
    desc={'summary':'A sufficiently long accurate summary of this object.',
          'materials':[{'name':'stonebrick'},'glass'],'dimensions_estimate':{'width':2,'depth':1,'height':1},
          'elements':['wall','window'],'rebuild_hints':['one','two']}
    return desc,grid,bbox


def test_known_perfect_and_fixed_composite(inputs):
    before=copy.deepcopy(inputs[0]); r=scorer.score_record(*inputs)
    assert r['scores']=={k:1 for k in [*scorer.WEIGHTS,'composite']}
    assert inputs[0]==before


@pytest.mark.parametrize('bad',[None,{}, {'width':True,'depth':'1','height':None}, {'width':0,'depth':float('inf'),'height':-1}, {'width':10**1000,'depth':Decimal('NaN'),'height':Decimal('Infinity')}])
def test_missing_dimensions_never_borrow_gt(inputs,bad):
    d,g,b=inputs; d['dimensions_estimate']=bad
    r=scorer.score_record(d,g,b)
    assert r['scores']['dimension_accuracy']==0
    assert not r['completeness_fields']['dimensions']
    assert all(x['predicted'] is None for x in r['dimensions'].values())


def test_partial_dimension_missing_penalty_and_decimal_rounding(inputs):
    d,g,b=inputs; d['dimensions_estimate']={'width':Decimal('2.5'),'height':1}
    r=scorer.score_record(d,g,b)
    assert r['scores']['dimension_accuracy']==pytest.approx(2/3)
    assert r['dimensions']['width']['predicted']==2
    assert r['dimensions']['depth']['predicted'] is None


def test_exact_ids_not_legacy_synonyms_and_coarse_is_separate(inputs):
    d,g,b=inputs; d['materials']=['stone_bricks','glass']
    s=scorer.score_record(d,g,b)['scores']
    assert s['strict_material_f1']==.5 and s['coarse_material_f1']==1


def test_ambiguous_invalid_materials_penalized_not_dropped(inputs):
    d,g,b=inputs; d['materials']=['stonebrick','glass','stone_or_glass',None,{'name':'stone | glass'}]
    r=scorer.score_record(d,g,b)
    assert r['scores']['strict_material_f1']==pytest.approx(4/7)
    assert r['scores']['coarse_material_f1']==pytest.approx(4/7)
    assert r['material_input_status']['invalid_count']==3
    assert scorer.coarse_token('nonsense')!=scorer.coarse_token('glowstone')


def test_air_exclusion_and_invalid_list(inputs):
    d,g,b=inputs; d['materials']=['minecraft:air','stonebrick','glass']
    assert scorer.score_record(d,g,b)['scores']['strict_material_f1']==1
    d['materials']='stonebrick'
    assert scorer.score_record(d,g,b)['scores']['strict_material_f1']==0


def test_completeness_rejects_wrong_typed_content(inputs):
    d,g,b=inputs; d.update(summary=['x']*40,elements=[None,{}],rebuild_hints=['',False])
    assert scorer.score_record(d,g,b)['scores']['completeness']==.4


def test_gt_target_frequency_lexical_ties_and_bbox_frame():
    g=Grid({(i,0,0):v for i,v in enumerate(['z','a','b','c','d','e','f'])})
    target,_=scorer.gt_material_target(g)
    assert target==set('abcdef')
    _,d=scorer.dimension_values({'width':7,'depth':1,'height':1},{'xmin':-1,'xmax':7,'ymin':0,'ymax':0,'zmin':0,'zmax':0})
    assert d['width']['gt']==9


def test_source_cardinality_and_corruption_fail_before_loading(tmp_path):
    with pytest.raises(ValueError,match='exact180'): scorer.compute_scores([],definition_sha256='x')
    p=tmp_path/'source.json'; p.write_text('{}')
    def forbidden(*args): raise AssertionError('must verify before loading')
    with pytest.raises(ValueError,match='source hash'):
        scorer.compute_scores([{'source_path':str(p),'source_sha256':'incorrect'}],definition_sha256='x',source_loader=forbidden,fixed_design=False)
