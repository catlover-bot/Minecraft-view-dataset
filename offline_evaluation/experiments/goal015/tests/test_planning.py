import copy
from decimal import Decimal
import hashlib
import json

import pytest

from offline_evaluation.experiments.goal015 import planning as p


def frame():
    return [{'scene_id':f'fixture_scene_{s}_{i:02d}', 'split':'evaluation', 'size_rank_stratum':s}
            for s in range(5) for i in range(12)]


def original():
    return {'summary':' Exact original.  未確定\n', 'building_type':'unknown',
            'shape':{'floors':None,'symmetric':False}, 'dimensions_estimate':{'w':1,'h':Decimal('1.00')},
            'materials':[{'name':'maybe wood','confidence':Decimal('0.90')}, {'name':'unknown'}],
            'elements':['last','first'], 'rebuild_hints':['do not assume'], 'uncertainties':['possibly wrong'],
            'provider':'private-provider', 'model':'private-model','building':'private-scene','created_at':'private-time'}


def test_selection_independent_hash_reference_and_no_outcome_dependency():
    f=frame();selected=p.select_scenes(f)
    actual={r['scene_id'] for r in selected}
    expected=set()
    for s in range(5):
        ids=[r['scene_id'] for r in f if r['size_rank_stratum']==s]
        expected.update(sorted(ids,key=lambda x:(hashlib.sha256(b'goal015-scenes-v1\0'+x.encode()).hexdigest(),x))[:2])
    assert actual==expected
    for row in f:row.update(iou='NEVER_READ',clipscore=object(),failure=True)
    assert p.select_scenes(f[::-1])==selected
    assert [{r['size_rank_stratum'] for r in selected[i:i+5]} for i in (0,5)]==[set(range(5))]*2


def test_schedule_balance_adjacent_reversal_builder_counterbalance():
    rows=p.schedule(p.select_scenes(frame()))
    assert p.validate_schedule(rows)
    assert len({r['paired_block_id'] for r in rows})==40
    for s in {r['scene_id'] for r in rows}:
        a=[r['builder'] for r in rows if r['scene_id']==s and r['repeat_index']==1][::2]
        b=[r['builder'] for r in rows if r['scene_id']==s and r['repeat_index']==2][::2]
        assert a==b[::-1]
    assert p.schedule(p.select_scenes(frame()))==rows
    rows[-1]=copy.deepcopy(rows[0])
    with pytest.raises(ValueError):p.validate_schedule(rows)


def test_exact_summary_typed_decimal_order_uncertainties_and_metadata_removed():
    value=original();raw=p._serialize_exact(value).encode()
    outputs,evidence=p.extract_conditions(raw)
    s=p.strict_saved_json(outputs['SUMMARY_ONLY'].decode());f=p.strict_saved_json(outputs['FULL_RECORD'].decode())
    assert s=={'summary':value['summary']} and f['summary']==value['summary']
    assert p.typed_tree(f)==p.typed_tree({k:value[k] for k in p.FIELDS})
    assert p.sha(s['summary'].encode())==p.sha(f['summary'].encode())==evidence['summary_sha256']
    assert f['elements']==['last','first'] and f['uncertainties']==['possibly wrong']
    assert type(f['dimensions_estimate']['w']) is int
    assert f['dimensions_estimate']['h'].as_tuple()==Decimal('1.00').as_tuple()
    assert all(b'private-' not in v for v in outputs.values())
    assert evidence['excluded_management_fields']==sorted(p.METADATA)


@pytest.mark.parametrize('key',['GT','scene_id','condition','repeat','unknown_semantic_field'])
def test_unclassified_top_level_rejected_before_payload(key):
    value=original();value[key]='do not drop silently'
    with pytest.raises(ValueError,match='unclassified'):p.extract_conditions(p._serialize_exact(value).encode())


@pytest.mark.parametrize('value',[None,False,0,'',' \t\n'])
def test_missing_empty_or_nonstring_summary_not_imputed(value):
    record=original();record['summary']=value
    with pytest.raises(ValueError):p.extract_conditions(p._serialize_exact(record).encode())


def test_missing_field_and_duplicate_json_rejected():
    value=original();del value['uncertainties']
    with pytest.raises(ValueError):p.extract_conditions(p._serialize_exact(value).encode())
    with pytest.raises(Exception):p.extract_conditions(b'{"summary":"a","summary":"b"}')
