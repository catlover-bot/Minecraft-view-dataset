"""Fabricated cohorts/files only; no research-source audit or model calls."""
from copy import deepcopy
from pathlib import Path
import json
import socket

import pytest

from offline_evaluation.analysis.goal011 import rest as r
from offline_evaluation.server_v2.cohort_v4 import freeze_ranked_split, canonicalize_saved_text


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw if isinstance(raw, bytes) else raw.encode())


def json_file(path, value): write(path, json.dumps(value, sort_keys=True, ensure_ascii=False))


def source(root, rel, kind, raw):
    path=root/rel; write(path, raw)
    return {'path':rel,'kind':kind,'sha256':r._hash(path),'size_bytes':path.stat().st_size}


def seal(package, root):
    c=package['cohort']; c.pop('cohort_sha256',None); c['cohort_sha256']=r._digest(c)
    p=Path(package['full_manifest']['identity']['cohort_path']);json_file(p,c)
    package['full_manifest']['identity']['cohort_file_sha256']=r._hash(p)
    return package


@pytest.fixture
def fabricated(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    rows=[{'scene_id':f'buildings_100_v4_building_{i:03d}','gt_occupied':i+1} for i in range(100)]
    c=freeze_ranked_split(rows);c.update(population_scenes=100,selection_uses_model_results=False)
    for row in c['scenes']:
        row.update(source_files=[],description_sources={},control={'reference_actions_json_bytes':row['gt_occupied']*50})
        if row['split']!='rest':continue
        b=row['scene_id'][-12:]; directory='datasets/buildings_100_v4/'+b;row['scene_path']=directory
        metadata = {'fixture': True, 'views': [{'path': f'images/v{k:02d}.png'} for k in range(12)]}
        row['source_files']=[source(root,directory+'/gt/voxels.npy','GT',('synthetic-npy-'+b).encode()),source(root,directory+'/gt/bbox.json','GT_bbox','{"xmin":0}'),source(root,directory+'/meta.json','scene_metadata',json.dumps(metadata))]
        for k in range(12):row['source_files'].append(source(root,directory+f'/images/v{k:02d}.png','image_reference',b'fixture-image'))
        for provider in r.PROVIDERS:
            text=json.dumps({'scene':row['scene_id'],'building':b,'provider':provider,'model':provider+'-fixture-model','description':'構造\r\nfixture','dimensions_estimate':{'width':3,'depth':2,'height':4}},ensure_ascii=False)
            s=source(root,f'outputs/i2t2b/buildings_100_v4/{b}/description_{provider}/description.json','saved_description',text)
            _,proof=canonicalize_saved_text(text);s.update(proof,status='prepared');row['description_sources'][provider]=s;row['source_files'].append(s)
        row['source_manifest_sha256']=r._digest(row['source_files'])
    package={'cohort':c,'full_manifest':{'identity':{'cohort_path':str(root/'frozen_run/cohort.json'),'evaluation_scene_ids':[x['scene_id'] for x in c['scenes'] if x['split']=='evaluation']}},'plans':{}}
    for label in ('parent160','original781','supplement3','supplement17'):
        d=root/'bound_runs'/label;d.mkdir(parents=True)
        package['plans'][label]={'identity':{'enabled_directory':str(d),'schedule':[{'scene_id':x} for x in package['full_manifest']['identity']['evaluation_scene_ids']]}}
    package['legacy_analysis_manifests']=[]
    seal(package,root)
    gate={'status':'PASS','input_kind':'fixture','phase0_binding':'PASS','definitions_frozen':True,'question_register_sha256':'a'*64,'analysis_plan_sha256':'b'*64,'code_commit':'c'*40}
    return root,package,gate


def run(fabricated,**kwargs):
    root,package,gate=fabricated
    return r.audit_rest30(package,gate=gate,root=root,**kwargs)


@pytest.mark.parametrize('field,value',[('status','FAIL'),('definitions_frozen',False),('analysis_plan_sha256',''),('question_register_sha256','a'),('code_commit','c'),('input_kind','unknown')])
def test_gate_fails_before_any_source_read(fabricated,monkeypatch,field,value):
    root,package,gate=fabricated;gate[field]=value
    monkeypatch.setattr(r,'_record',lambda *a,**kw:pytest.fail('gate must precede reads'))
    with pytest.raises(r.RestAuditError):run(fabricated)


def test_full_fixture_30_scenes90_inputs_no_network_or_mutation(fabricated,monkeypatch):
    root,package,gate=fabricated;before=deepcopy(package)
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**kw:pytest.fail('no network'))
    result=run(fabricated);f=result['feasibility'];rows=result['sources']
    assert package==before and len(rows)==90 and len({x['scene_id'] for x in rows})==30
    assert f['rest_per_size_quintile']=={str(q):6 for q in range(5)}
    assert f['provider_source_counts']=={p:30 for p in r.PROVIDERS}
    assert f['verified_selected_inputs']==30 and f['not_evaluated_in_verified_fixed_plans'] is True
    assert f['truly_unseen_anywhere']=='NOT_VERIFIED' and f['future_protocol_status']=='UNAPPROVED_NOT_ENABLED'
    assert f['source_before_after_unchanged'] and f['research_api_requests_sent']==0
    assert f['source_writes']==0 and f['api_key_queries']==0
    assert all(x['input_utf8_bytes']>x['input_characters'] for x in rows)
    assert all(x['saved_image_count']==12 and x['actual_description_request_view_count'] is None for x in rows)
    assert all(x['future_context_fit_status']=='NOT_VERIFIED' for x in rows)
    r.verify_inventory(result['inventory'])


@pytest.mark.parametrize('kind',['duplicate','stratum','split','rank'])
def test_invalid_cohort_assignment_rejected(fabricated,kind):
    root,package,_=fabricated;cohort=package['cohort']
    if kind=='duplicate':cohort['scenes'][0]=deepcopy(cohort['scenes'][1])
    elif kind=='stratum':cohort['scenes'][0]['size_rank_stratum']=4
    elif kind=='rank':cohort['scenes'][0]['rank']+=1
    else:cohort['scenes'][0]['split']='rest'
    seal(package,root)
    with pytest.raises(r.RestAuditError):run(fabricated)


def test_source_hash_mismatch_is_not_missing_data(fabricated):
    root,package,_=fabricated;scene=next(x for x in package['cohort']['scenes'] if x['split']=='rest')
    write(root/scene['source_files'][0]['path'],b'tampered')
    with pytest.raises(r.RestAuditError,match='mismatch'):run(fabricated)


def test_description_absence_retained_no_provider_switch(fabricated):
    root,package,_=fabricated;scene=next(x for x in package['cohort']['scenes'] if x['split']=='rest')
    scene['description_sources']['openai']={'status':'infeasible','reason':'fixture absent'}
    seal(package,root);out=run(fabricated)
    assert out['feasibility']['verified_selected_inputs']==29 and len(out['sources'])==90
    row=next(x for x in out['sources'] if x['scene_id']==scene['scene_id'] and x['description_provider']=='openai')
    assert row['source_status']=='NOT_AVAILABLE' and row['input_path'] is None
    assert row['selected_for_prospective_design'] is True


@pytest.mark.parametrize('change',['traversal','symlink','duplicate_image','typed_tree','description_descriptor'])
def test_bad_source_provenance_stops(fabricated,change):
    root,package,_=fabricated;scene=next(x for x in package['cohort']['scenes'] if x['split']=='rest')
    if change=='traversal':scene['source_files'][0]['path']='../private/voxels.npy'
    elif change=='symlink':
        p=root/scene['source_files'][0]['path'];p.unlink();p.symlink_to(root/'does-not-exist')
    elif change=='duplicate_image':scene['source_files'].append(deepcopy(scene['source_files'][3]))
    elif change=='description_descriptor':
        scene['description_sources']['openai']=dict(scene['description_sources']['openai'],sha256='0'*64)
    else:scene['description_sources']['openai']['typed_tree_sha256']='0'*64
    scene['source_manifest_sha256']=r._digest(scene['source_files']);seal(package,root)
    with pytest.raises((r.RestAuditError,FileNotFoundError)):run(fabricated)


def test_prior_plan_exposure_preserved_and_incomplete_scope_not_certified(fabricated):
    root,package,_=fabricated;sid=next(x['scene_id'] for x in package['cohort']['scenes'] if x['split']=='rest')
    package['plans']['parent160']['identity']['schedule'].append({'scene_id':sid})
    a=run(fabricated)['feasibility'];assert a['rest_scenes_found_in_verified_fixed_plans']==[sid] and a['not_evaluated_in_verified_fixed_plans'] is False
    package['plans'].pop('supplement17');b=run(fabricated)['feasibility']
    assert b['not_evaluated_in_verified_fixed_plans'] is None and b['verified_fixed_plan_scope_complete'] is False


def test_qualified_prior_v5_artifact_and_ambiguous_other_dataset_separated(fabricated):
    root,package,_=fabricated;sid=next(x['scene_id'] for x in package['cohort']['scenes'] if x['split']=='rest');b=sid[-12:]
    exact=root/f'outputs/i2t2b/buildings_100_v4/{b}/rebuild_world_schema_material_v5/voxels.npy';write(exact,b'prior-output')
    other=root/f'outputs/i2t2b/buildings_100_v1/{b}/rebuild_world_schema_material_v5/voxels.npy';write(other,b'other-output')
    human=root/'outputs/human_rebuild/cases.json';json_file(human,{'scene_id':sid})
    figures=root/'reports/figures/past.svg';write(figures,'<svg>'+b+'</svg>')
    hits=run(fabricated)['feasibility']['prior_exposure_audit']['hits']
    a=[x for x in hits if x['path']==str(exact)];o=[x for x in hits if x['path']==str(other)]
    assert any(x['references_exact_v4_scene_identifier'] and x['schema_v5_named_in_path'] for x in a)
    assert all(not x['references_exact_v4_scene_identifier'] for x in o)
    assert any(x['path']==str(human) and x['kind'].startswith('HUMAN_STUDY') for x in hits)
    assert any(x['path']==str(figures) and x['identity_match']=='AMBIGUOUS_BASENAME_ONLY' for x in hits)
    assert all(not x['model_success_or_completed_human_experiment_inferred'] for x in hits)
    assert all('excerpt' not in x and 'content' not in x for x in hits)


def test_does_not_search_auth_home_new_goal_or_symlinks(fabricated):
    root,package,_=fabricated;sid=next(x['scene_id'] for x in package['cohort']['scenes'] if x['split']=='rest')
    for p in [root/'reports/goal011/a.json',root/'reports/.env',root/'outputs/.ssh/id.key',root/'outputs/authority.json']:
        write(p,sid)
    outside=root.parent/'outside.txt';write(outside,sid);(root/'reports/escape.txt').symlink_to(outside)
    audit=run(fabricated)['feasibility']['prior_exposure_audit']
    assert not any(x['path'] in {str(outside),str(root/'reports/escape.txt')} or '.env' in x['path'] or 'goal011' in Path(x['path']).parts for x in audit['hits'])
    package['legacy_analysis_manifests']=[{'canonical_run_root':'/root/.ssh'}]
    with pytest.raises(r.RestAuditError,match='external scope'):run(fabricated)


def test_nonunique_gt_hash_is_not_identity_proof():
    sid='buildings_100_v4_building_001';h='a'*64
    got=r._matches(h.encode(),Path('/tmp/result.json'),{sid},{h:{sid}})
    assert (sid,'GT_CONTENT_HASH_REFERENCE_NOT_UNIQUE_SCENE_PROOF') in got
    assert not r._classification('/tmp/result.json',next(iter(got))[1])['references_exact_v4_scene_identifier']


def test_binary_and_resource_limits_explicit_no_false_unseen(fabricated):
    root,package,_=fabricated;write(root/'reports/figure.png',b'opaque-content')
    f=run(fabricated,limits=r.RestLimits(max_search_bytes=1))['feasibility']
    skipped=f['prior_exposure_audit']['not_searched']
    assert any(x['reason']=='SEARCH_RESOURCE_LIMIT' for x in skipped)
    assert any(x['reason']=='BINARY_CONTENT_NOT_SEARCHED' for x in skipped)
    assert f['truly_unseen_anywhere']=='NOT_VERIFIED'


def test_candidate_enumeration_limit_fails_closed(fabricated):
    with pytest.raises(r.RestAuditError,match='inventory resource limit'):
        run(fabricated,limits=r.RestLimits(max_search_files=2))


def test_frozen_plan_caps_and_explicit_missing_scope(fabricated):
    assert r.RestLimits().max_search_files == 100_000
    assert r.RestLimits().max_search_bytes == 1024**3
    assert r.RestLimits().max_text_file_bytes == 32*1024**2
    audit=run(fabricated)['feasibility']['prior_exposure_audit']
    omitted={Path(x['path']).name:x for x in audit['excluded'] if x['reason']=='SCOPE_DIRECTORY_ABSENT'}
    assert {'experiments','figures','human_studies'} <= set(omitted)
    assert all(x['evidence_status']=='NOT_VERIFIED' for x in audit['excluded']+audit['not_searched'])


def test_evidence_cap_stops_without_positive_truncated_report(fabricated):
    with pytest.raises(r.RestAuditError,match='evidence limit'):
        run(fabricated,limits=r.RestLimits(max_evidence_rows=1))


def test_change_after_scanning_fails_inventory_check(fabricated,monkeypatch):
    root,package,_=fabricated;original=r.audit_exposures
    def mutate(*args,**kwargs):
        result=original(*args,**kwargs);scene=next(x for x in package['cohort']['scenes'] if x['split']=='rest')
        write(root/scene['source_files'][0]['path'],b'changed-after-search');return result
    monkeypatch.setattr(r,'audit_exposures',mutate)
    with pytest.raises(r.RestAuditError,match='inventory changed'):run(fabricated)


def test_chunk_boundary_scene_id_search(fabricated):
    root,package,_=fabricated;sid=next(x['scene_id'] for x in package['cohort']['scenes'] if x['split']=='rest')
    p=root/'reports/split.txt';write(p,b' '*(1024*1024-9)+sid.encode()+b'\n')
    hits=run(fabricated)['feasibility']['prior_exposure_audit']['hits']
    assert any(x['path']==str(p) and x['scene_id']==sid and x['identity_match']=='EXACT_V4_SCENE_ID' for x in hits)


@pytest.mark.parametrize('kind', ['excluded_root', 'symlink_root'])
def test_manifest_bound_scope_cannot_bypass_root_exclusions(fabricated, monkeypatch, kind):
    root, package, _ = fabricated
    if kind == 'excluded_root':
        unsafe = root/'outputs/.aws'
        unsafe.mkdir(parents=True)
    else:
        unsafe = root/'scope-link'
        unsafe.symlink_to(root/'outputs', target_is_directory=True)
    package['plans']['parent160']['identity']['enabled_directory'] = str(unsafe)
    monkeypatch.setattr(r, '_record', lambda *a, **kw: pytest.fail('unsafe scope must fail before source reads'))
    with pytest.raises(r.RestAuditError): run(fabricated)


def saved_request(fabricated):
    root, package, _ = fabricated
    row = next(x for x in package['cohort']['scenes'] if x['split'] == 'rest')
    path = (root/row['description_sources']['openai']['path']).with_name('description.request.json')
    value = {'building': row['scene_id'][-12:], 'provider': 'openai', 'model': 'openai-fixture-model',
             'images': ['images/v05.png', 'images/v01.png'], 'prompt': 'PRIVATE_FIXTURE_PROMPT'}
    json_file(path, value)
    return row, path, value


def test_saved_request_preserves_order_and_binds_each_image_without_transport_claim(fabricated):
    row, path, _ = saved_request(fabricated)
    result = run(fabricated)
    item = next(x for x in result['sources'] if x['scene_id'] == row['scene_id'] and x['description_provider'] == 'openai')
    assert item['description_request_sha256'] == r._hash(path)
    assert item['request_view_count_status'] == 'VERIFIED_SAVED_REQUEST_IMAGE_BINDINGS'
    assert item['actual_description_request_view_count'] == 2
    assert [x['meta_view_index'] for x in item['request_images_ordered']] == [5, 1]
    assert [x['request_order'] for x in item['request_images_ordered']] == [0, 1]
    assert item['request_submission_independently_verified'] is False
    assert result['feasibility']['verified_saved_request_image_binding_counts'] == {'openai': 1, 'anthropic': 0, 'gemini': 0}
    for image in item['request_images_ordered']:
        assert result['inventory'][image['path']]['sha256'] == image['sha256']
    assert 'PRIVATE_FIXTURE_PROMPT' not in json.dumps(result)
    assert 'prior_exposure_evidence' not in item
    evidence = [x for x in result['feasibility']['prior_exposure_audit']['hits'] if x['scene_id'] == row['scene_id']]
    assert item['prior_exposure_evidence_sha256'] == r._digest(evidence)


@pytest.mark.parametrize('change', ['building', 'provider', 'model', 'duplicate_image', 'unknown_image', 'traversal', 'symlink'])
def test_saved_request_identity_and_image_scope_fail_closed(fabricated, change):
    _, path, value = saved_request(fabricated)
    if change in ('building', 'provider', 'model'): value[change] = 'different'
    elif change == 'duplicate_image': value['images'].append(value['images'][0])
    elif change == 'unknown_image': value['images'][0] = 'images/absent.png'
    elif change == 'traversal': value['images'][0] = '../outside.png'
    else:
        target = path.with_name('other-request.json'); path.rename(target); path.symlink_to(target)
    if change != 'symlink': json_file(path, value)
    with pytest.raises(r.RestAuditError): run(fabricated)


def test_request_change_after_search_fails_source_inventory(fabricated, monkeypatch):
    _, path, value = saved_request(fabricated)
    original = r.audit_exposures
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        value['model'] = 'changed-after-search'; json_file(path, value)
        return result
    monkeypatch.setattr(r, 'audit_exposures', mutate)
    with pytest.raises(r.RestAuditError, match='inventory changed'): run(fabricated)


def test_previous_offline_analysis_is_in_search_scope_but_new_goal_is_not(fabricated):
    root, package, _ = fabricated
    sid = next(x['scene_id'] for x in package['cohort']['scenes'] if x['split'] == 'rest')
    previous = root/'offline_evaluation/analysis/goal010/old_analysis.py'
    current = root/'offline_evaluation/analysis/goal011/new_analysis.py'
    write(previous, '# fixture scene reference '+sid)
    write(current, '# current audit is not previous exposure '+sid)
    hits = run(fabricated)['feasibility']['prior_exposure_audit']['hits']
    assert any(h['path'] == str(previous) for h in hits)
    assert not any(h['path'] == str(current) for h in hits)
