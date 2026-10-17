"""Synthetic identity/missingness fixtures; no original artifact access."""
import copy
import hashlib
import pytest
from offline_evaluation.analysis.goal014.binding import bind_rows, object_hash


def fixture():
    sha = 'a'*64
    gt = dict(grid_sha256=sha, voxels_sha256='b'*64, bbox_sha256='c'*64)
    plan = {'identity': {'gt_by_scene': {'scene': gt}}}
    plan['identity_hash'] = object_hash(plan['identity'], formatted=True)
    rows, scores, slots, metrics = [], [], [], []
    row = dict(scene_id='scene', description_provider='gemini', score=.4,
               score_status='COMPUTATION_VALID', summary_token_count_untruncated=10,
               O_input_id='input', O_input_sha256='d'*64, outcomes={})
    score = dict(row, token_diagnostics=dict(token_coverage_status='VERIFIED', phase2_token_eligible=True,
                  truncated=False, summary_dropped_tokens=0, summary_token_count_untruncated=10))
    for i, builder in enumerate(['openai','anthropic']):
        original = {'alignment': {'shift_xyz':[1,2,3]}, 'aligned': {'occupancy_iou': .3}}
        s=dict(global_schedule_index=i, scene_id='scene', description_model='gemini', provider=builder,
            condition='O', input_id='input',input_sha256='d'*64, attempt_id=str(i),
            original_attempt_id=str(i), selection_reason='original_first_attempt', source_label='source',
            source_plan_identity=plan['identity_hash'],source_entry_hash='e'*64,
            source_gt_descriptor_hash=object_hash(gt,formatted=True),gt_grid_sha256=sha,status='success',
            failure_kind=None,usage_known=True,completed_request_outcome=True,
            selected_attempt_id=str(i),selected_final_observation=True,artifacts={'voxels.json':{'sha256':'f'*64}},
            metrics=original)
        p=dict(gt_grid_sha256='1'*64,pred_grid_sha256='2'*64,shift_sha256=object_hash([1,2,3]),
               stored_alignment_shift_xyz=[1,2,3],alignment_searched=False,failure0_applied=False,
               protocol_id='goal010v2_fixed_shift_metrics_v1',alignment_protocol='server_v2.2_strict_translation',
               implementation_sha256='3'*64)
        p['inputs_and_shift_sha256']=hashlib.sha256(':'.join(p[k] for k in ['gt_grid_sha256','pred_grid_sha256','shift_sha256']).encode()).hexdigest()
        m=dict(s,artifact_sha256={'voxels.json':'f'*64},original_metric_object_sha256=object_hash(original),
               gt_voxels_sha256=gt['voxels_sha256'],gt_bbox_sha256=gt['bbox_sha256'],
               metrics={'aligned':{'surface_f1_at_1':.6,'occupancy_iou':.3}},null_reasons={'aligned':{}},
               metric_provenance=p,metric_status='COMPUTED',shift_xyz=[1,2,3],
               implementation_hashes={'offline_evaluation/analysis/goal010v2/metrics.py':'3'*64})
        row['outcomes'][builder]=dict(global_schedule_index=i,selected_attempt_id=str(i),aligned_occupancy_iou=.3,status='success')
        slots.append(s);metrics.append(m)
    return [row],[score],slots,metrics,{'source':plan}


def test_join_preserves_inputs_and_outcome_shape():
    args=fixture(); before=copy.deepcopy(args)
    rows=bind_rows(*args,fixture=True)
    assert args==before
    assert rows[0]['truncated'] is False
    assert rows[0]['outcomes']['openai']['aligned_surface_f1_at_1']==.6
    assert rows[0]['outcomes']['anthropic']['aligned_occupancy_iou']==.3


@pytest.mark.parametrize('field,value', [('scene_id','other'),('description_model','other'),('provider','other'),
    ('input_id','other'),('input_sha256','0'*64),('attempt_id','other'),('source_plan_identity','0'*64),
    ('source_entry_hash','0'*64),('source_gt_descriptor_hash','0'*64),('gt_grid_sha256','0'*64),
    ('gt_voxels_sha256','0'*64),('gt_bbox_sha256','0'*64),('original_metric_object_sha256','0'*64)])
def test_reject_identity_mismatch(field,value):
    args=fixture();args[3][0][field]=value
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


@pytest.mark.parametrize('field,value',[('shift_sha256','0'*64),('pred_grid_sha256','0'*64),
    ('gt_grid_sha256','0'*64),('inputs_and_shift_sha256','0'*64),('stored_alignment_shift_xyz',[1,2,4]),
    ('alignment_searched',True),('failure0_applied',True),('implementation_sha256','0'*64)])
def test_reject_engine_binding(field,value):
    args=fixture();args[3][0]['metric_provenance'][field]=value
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


def test_reject_selected_attempt():
    args=fixture();args[2][0]['selected_attempt_id']='other'
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


def test_reject_prediction_source_hash():
    args=fixture();args[3][0]['artifact_sha256']['voxels.json']='0'*64
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,2,True])
def test_reject_nonfinite_surface(value):
    args=fixture();args[3][0]['metrics']['aligned']['surface_f1_at_1']=value
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


def test_null_surface_retains_reason():
    args=fixture();args[3][0]['metrics']['aligned']['surface_f1_at_1']=None
    args[3][0]['null_reasons']['aligned']['surface_f1_at_1']='FIXTURE_RESOURCE_LIMIT'
    out=bind_rows(*args,fixture=True)[0]['outcomes']['openai']
    assert out['aligned_surface_f1_at_1'] is None
    assert out['surface_null_reason']=='FIXTURE_RESOURCE_LIMIT'


def test_null_surface_requires_reason():
    args=fixture();args[3][0]['metrics']['aligned']['surface_f1_at_1']=None
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


@pytest.mark.parametrize('flag',[None,1,'False',True])
def test_bad_truncation_flag(flag):
    args=fixture();args[1][0]['token_diagnostics']['truncated']=flag
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


def test_empty_summary_preserved():
    args=fixture()
    for row in [args[0][0],args[1][0]]:row.update(score=None,score_status='MISSING_EMPTY_SUMMARY')
    args[1][0]['token_diagnostics']={}
    out=bind_rows(*args,fixture=True)[0]
    assert out['score'] is None and out['truncated'] is None


def test_reject_duplicate_and_missing_metric_rows():
    args=fixture();args[3].append(copy.deepcopy(args[3][0]))
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)
    args=fixture();args[3].pop()
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)


def test_frozen_score_not_replaced():
    args=fixture();args[0][0]['score']=.8
    with pytest.raises(ValueError):bind_rows(*args,fixture=True)
