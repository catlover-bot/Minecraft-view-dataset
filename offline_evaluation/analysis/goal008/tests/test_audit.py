from copy import deepcopy
import json
from pathlib import Path
import pytest
from offline_evaluation.analysis.goal008 import audit


@pytest.fixture
def binding(tmp_path):
    source_path=tmp_path/'description.json'
    source_path.write_text('{\n"summary":"A small stone building", "dimensions_estimate":{"width":1.0,"height":2,"depth":3}, "materials":[{"name":"stone"}], "provider":"openai", "model":"fixture-description-model"\n}\n')
    text=source_path.read_text(); canonical,proof=audit.canonicalize_saved_text(text)
    source={'path':'description.json','sha256':audit.sha256_file(source_path),**proof}
    gt_dir=tmp_path/'building';(gt_dir/'gt').mkdir(parents=True)
    (gt_dir/'gt/bbox.json').write_text('{}');(gt_dir/'gt/voxels.npy').write_bytes(b'fixture-not-loaded')
    gt={'building_directory':str(gt_dir),'bbox_sha256':audit.sha256_file(gt_dir/'gt/bbox.json'),
        'voxels_sha256':audit.sha256_file(gt_dir/'gt/voxels.npy'),'grid_sha256':'fixture-grid'}
    originals=[];jrows=[];slots=[]
    for index,provider in enumerate(audit.BUILDERS):
        common={'scene_id':'fixture_scene','description_model':'openai','provider':provider,'source_sha256':source['sha256']}
        originals.append({**common,'condition':'O','input_text':text,'input_id':'O-input','input_sha256':audit.digest(text.encode()),'schedule_index':index})
        jrows.append({**common,'condition':'J','input_text':canonical,'input_id':'J-input','input_sha256':audit.digest(canonical.encode()),'schedule_index':index+2})
        slots.append({'global_schedule_index':index,'condition':'O','provider':provider,'description_model':'openai','scene_id':'fixture_scene',
            'input_id':'O-input','input_sha256':audit.digest(text.encode()),'gt_grid_sha256':'fixture-grid','attempt_id':'attempt-'+provider,
            'status':'success','source_label':'fixture','source_plan_identity':'fixture-plan','completed_request_outcome':True,
            'usage_known':True,'builder_model':'fixture-builder-'+provider})
    return {'scene':'fixture_scene','provider':'openai','source':source,'gt':gt,'original_rows':originals,'j_rows':jrows,'slots':slots,'root':tmp_path}


def test_one_description_binds_two_builders_without_pseudo_replication(binding):
    row=audit.bind_description(**binding)
    assert row['description_provider']=='openai'
    assert row['description_model_id']=='fixture-description-model'
    assert row['input_id']=='O-input' and row['j_input_id']=='J-input'
    assert set(row['builder_observations'])==set(audit.BUILDERS)
    assert row['source_sha256']==row['input_sha256']
    assert row['O_J_typed_tree_equal'] is True
    assert row['score_status']=='NOT_COMPUTED_BY_SOURCE_AUDIT'
    assert 'input_text' not in row and 'summary' not in row


@pytest.mark.parametrize('mutation', ['source_bytes','typed_tree','J_number_type','J_whitespace','O_bytes','provider_repetition','semantic_id','slot_input','slot_GT','GT_bytes','escape'])
def test_source_binding_rejects_mismatch(binding,mutation):
    b=binding
    if mutation=='source_bytes':(b['root']/'description.json').write_text('{}')
    elif mutation=='typed_tree':b['source']['typed_tree_sha256']='changed'
    elif mutation=='J_number_type':b['j_rows'][0]['input_text']=b['j_rows'][0]['input_text'].replace('1.0','1')
    elif mutation=='J_whitespace':b['j_rows'][0]['input_text']+=' '
    elif mutation=='O_bytes':b['original_rows'][0]['input_text']+=' '
    elif mutation=='provider_repetition':b['original_rows'][1]['provider']='openai'
    elif mutation=='semantic_id':b['original_rows'][1]['input_id']='other'
    elif mutation=='slot_input':b['slots'][0]['input_sha256']='other'
    elif mutation=='slot_GT':b['slots'][0]['gt_grid_sha256']='other'
    elif mutation=='GT_bytes':Path(b['gt']['building_directory'],'gt/bbox.json').write_text('{"changed":true}')
    else:b['source']['path']='../escape.json'
    with pytest.raises(audit.AuditError):audit.bind_description(**b)


def test_duplicate_keys_are_not_losslessly_normalized(binding):
    p=binding['root']/'description.json';p.write_text('{"provider":"openai","model":"fixture","width":1,"width":2}')
    binding['source']['sha256']=audit.sha256_file(p)
    with pytest.raises(ValueError,match='duplicate'):audit.bind_description(**binding)


def test_response_metadata_excludes_raw_content_ids_and_reasoning_text():
    raw={'temperature':1.0,'top_p':1.0,'seed':None,'model':'fixture-model','id':'private-id','output':[{'text':'private-content'}],
         'instructions':'private-prompt','api_key':'private-key','reasoning':{'effort':'medium','mode':'standard','summary':'private-reasoning',
         'encrypted_content':'private-ciphertext'},'headers':{'Authorization':'private-auth'}}
    result=audit.safe_response_sampling(raw)
    assert result=={'temperature':1.0,'top_p':1.0,'seed':None,'model':'fixture-model','reasoning':{'effort':'medium','mode':'standard'}}
    assert 'private-' not in json.dumps(result)


def test_unreported_sampling_parameters_remain_absent():
    assert audit.safe_response_sampling({'model':'anthropic-fixture','content':[]})=={'model':'anthropic-fixture'}


def test_nested_values_cannot_be_smuggled_via_sampling_fields():
    assert audit.safe_response_sampling({'temperature':{'private':'secret'},'model':['private'],'reasoning':'secret'})=={}


def test_failed_source_gate_cannot_enable_followup_scoring():
    with pytest.raises(audit.AuditError,match='not_passed'):
        audit.verify_report({'version':audit.VERSION,'status':'NOT_TESTABLE'})


def test_goal008_directory_is_outside_frozen_goal007_hash_scope():
    source=Path(audit.__file__).resolve()
    hashes=audit.old_cli.implementation_hashes()
    assert source.relative_to(audit.ROOT).as_posix() not in hashes
    assert not any('/goal008/' in name for name in hashes)
    frozen=audit.read_json(audit.GOAL007/'MANIFEST.json')['code_hashes']
    assert hashes==frozen


@pytest.fixture
def oj_plans(tmp_path):
    schedule=[]
    for scene in range(60):
        for desc in audit.DESCRIPTION_PROVIDERS:
            for provider in audit.BUILDERS:
                for condition in ('O','J'):
                    schedule.append({'scene_id':f'scene_{scene}','description_model':desc,'provider':provider,
                        'condition':condition,'model':audit.adapter.MODELS[provider],
                        'input_text':'{"fixture":true}','schedule_index':len(schedule)})
    # The one authorized repeated slot is O/OpenAI, not J.
    schedule[0]['schedule_index']=940
    identity={'schedule':schedule,'prompt':'fixture prompt','budgets':{'max_output_tokens':32768},'enabled_directory':str(tmp_path/'original')}
    entries=[]
    for index,r in enumerate(schedule):
        payload=audit.adapter.build_payload(r['provider'],r['model'],identity['prompt'],r['input_text'],32768)
        entries.append({'attempt_id':f'{index:032x}','request_payload_hash':audit.object_hash(payload),'artifacts':{},
            'retry':False,'usage':{} if index else None,'status':'success' if index else 'failed','http_status_code':None if index else 401})
    original_dir=tmp_path/'original';original_dir.mkdir();audit.write_new(original_dir/'ledger.json',{'entries':entries})
    retry_dir=tmp_path/'retry';retry_dir.mkdir()
    retry=deepcopy(entries[0]);retry.update(attempt_id='f'*32,retry=True,source_attempt_id=entries[0]['attempt_id'],status='success',usage={},http_status_code=None)
    audit.write_new(retry_dir/'ledger.json',{'entries':[retry]})
    return {'original781':{'identity_hash':'original-fixture','identity':identity},
            'supplement3':{'identity_hash':'retry-fixture','identity':{**identity,'schedule':[schedule[0]],'enabled_directory':str(retry_dir)}}}


def test_sampling_audit_counts_auth_retry_in_O_not_J(oj_plans):
    report=audit.sampling_audit(oj_plans)
    assert report['O_attempts']==361 and report['J_attempts']==360
    assert report['OJ_attempts']==721 and report['distinct_builder_description_condition_slots']==720
    assert report['per_slot_attempt_frequencies']=={'1':719,'2':1}
    assert [r['condition']for r in report['approved940_retry']]==['O','O']
    assert report['explicit_decoding_seed'] is None and report['controlled_sampling_replicates'] is False
    assert all(not r['explicit_sampling_parameters'] for r in report['attempts'])


def test_sampling_audit_rejects_falsely_linked_retry(oj_plans):
    path=Path(oj_plans['supplement3']['identity']['enabled_directory'])/'ledger.json'
    value=json.loads(path.read_text());value['entries'][0]['source_attempt_id']='wrong-origin'
    path.write_text(json.dumps(value))
    with pytest.raises(audit.AuditError,match='approved940_retry_changed'):audit.sampling_audit(oj_plans)
