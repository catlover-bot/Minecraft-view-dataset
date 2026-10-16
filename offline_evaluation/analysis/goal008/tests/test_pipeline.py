import copy
import pytest
from offline_evaluation.analysis.goal008 import pipeline


def test_commit_requires_full_pinned_sha():
    with pytest.raises(ValueError,match='explicit'): pipeline.verify_committed('HEAD')


def test_augmentation_requires_all_source_bindings_and_preserves_originals():
    gt={'grid_sha256':'g','bbox_sha256':'b','voxels_sha256':'v'}
    r={'condition':'O','scene_id':'s','description_model':'openai','provider':'openai','input_id':'i','input_sha256':'h',
       'gt_grid_sha256':'g','source_gt_descriptor_hash':'d','attempt_id':'a','global_schedule_index':0}
    source={'scene_id':'s','description_provider':'openai','input_id':'i','input_sha256':'h','source_sha256':'x',
        'gt_descriptor':gt,'gt_descriptor_sha256':'d','builder_observations':{'openai':{'attempt_id':'a','global_schedule_index':0}}}
    original=copy.deepcopy(r)
    result=pipeline.augment_slots([r],[source])
    assert r==original and result[0]['source_sha256']=='x'
    assert result[0]['gt_voxels_sha256']=='v'
    source['input_sha256']='wrong'
    with pytest.raises(ValueError,match='augmentation mismatch'):pipeline.augment_slots([r],[source])


def test_existing_output_stops_before_audit(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline.audit,'verify_report',lambda *a:pytest.fail('must stop before reads'))
    with pytest.raises(ValueError,match='new research output'):pipeline.run_phase1('unused',tmp_path,'HEAD')


def test_csv_exclusive_lf_null_and_finite(tmp_path):
    p=tmp_path/'a.csv';pipeline.write_csv(p,[{'a':None,'b':[1,2]}])
    assert b'\r' not in p.read_bytes() and p.read_text().startswith('a,b\n')
    with pytest.raises(FileExistsError):pipeline.write_csv(p,[])


def test_phase1_completion_keeps_real_json_path_boundaries_and_source_bytes(tmp_path,monkeypatch):
    """Orchestration fixture only: real JSON/file writes, no real scores or CI."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from contextlib import nullcontext
    actual_path=Path
    research=tmp_path/'research';research.mkdir();out=research/'phase1'
    monkeypatch.setattr(pipeline,'Path',lambda value:research if str(value)=='/root/minecraft-runs' else actual_path(value))
    old=tmp_path/'goal007';(old/'data').mkdir(parents=True);(old/'results').mkdir()
    report=tmp_path/'goal008';report.mkdir()
    for name in ('definition.md','stats_definition.md','example_definition.md'):(report/name).write_text('frozen fixture definition\n')
    monkeypatch.setattr(pipeline,'REPORT',report);monkeypatch.setattr(pipeline.audit,'GOAL007',old)
    gt={'grid_sha256':'g','bbox_sha256':'b','voxels_sha256':'v','building_directory':'fixture-only'}
    row={'condition':'O','scene_id':'s','description_model':'openai','provider':'openai','input_id':'i','input_sha256':'h',
       'gt_grid_sha256':'g','source_gt_descriptor_hash':'d','attempt_id':'a','global_schedule_index':0}
    source={'scene_id':'s','description_provider':'openai','description_model_id':'fixture-model','input_id':'i','input_sha256':'h',
        'j_input_id':'j','j_input_sha256':'jh','source_path':'fixture-only-not-loaded','source_sha256':'x',
        'gt_descriptor':gt,'gt_descriptor_sha256':'d','gt_bbox_path':'fixture-bbox','gt_voxels_path':'fixture-voxels',
        'builder_observations':{'openai':{'attempt_id':'a','global_schedule_index':0}}}
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps({'identity':{'fixture':True}}))
    (old/'MANIFEST.json').write_text(json.dumps({'sources':{'fixture':{'plan_path':str(plan_path)}}}))
    slot_path=old/'data/final_slots_960.json';slot_path.write_text(json.dumps([row]))
    (old/'results/comparisons_20.csv').write_text('id\nfixture\n')
    original_bytes={str(p):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    sources=tmp_path/'sources.json';sources.write_text(json.dumps([source]))
    gate={'status':'PASS','description_sources_path':str(sources)}
    gate_path=tmp_path/'source_gate.json';gate_path.write_text(json.dumps(gate))
    hashes={'fixture.py':'frozen-hash'};commit_calls=[]
    def verify_committed(commit):commit_calls.append(commit);return hashes
    monkeypatch.setattr(pipeline,'verify_committed',verify_committed)
    gate_calls=[]
    def verify_report(value):gate_calls.append(value);return gate
    monkeypatch.setattr(pipeline.audit,'verify_report',verify_report)
    monkeypatch.setattr(pipeline.audit.old_audit,'source_locks',lambda _:nullcontext())
    monkeypatch.setattr(pipeline.audit.old_audit,'process_audit',lambda:{})
    monkeypatch.setattr(pipeline.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    stats_hash=pipeline.sha256_file(report/'stats_definition.md')
    monkeypatch.setattr(pipeline.statistics,'definition_sha256',lambda:stats_hash)
    score={**source,'gt_bbox_sha256':'b','gt_voxels_sha256':'v','scoring_specification_id':'fixture-only','scorer_sha256':'fixture',
        'definition_sha256':pipeline.sha256_file(report/'definition.md'),'scores':{f:0.5 for f in pipeline.statistics.FEATURES},
        'missing_reasons':{},'invalid_material_tokens':[]}
    def compute_scores(source_rows,**kwargs):
        assert (out/'definitions_frozen.json').exists()
        assert source_rows==[source]
        assert kwargs['definition_sha256']==score['definition_sha256']
        return [score]
    monkeypatch.setattr(pipeline.scorer,'compute_scores',compute_scores)
    def analyze(scores,slots,**kwargs):
        assert slots[0]['source_sha256']=='x' and slots[0]['gt_voxels_sha256']=='v'
        assert kwargs['audit_gate']['definitions_frozen'] is True
        assert kwargs['definition_sha256']==stats_hash
        return {'associations':[{'status':'FIXTURE_ONLY','value':None}], 'status_counts':{'FIXTURE_ONLY':1},'joined_descriptions':[{'fixture':True}]}
    monkeypatch.setattr(pipeline.statistics,'analyze_correlations',analyze)
    monkeypatch.setattr(pipeline,'descriptive_scores',lambda _:[{'fixture':True}])
    # audit.read_json is intentionally real, accepting persisted str references.
    result=pipeline.run_phase1(str(gate_path),out,'a'*40)
    persisted=json.loads((out/'PHASE1_MANIFEST.json').read_text())
    assert persisted==result and result['status']=='PHASE0_PHASE1_PASS'
    assert result['source_gate_sha256']==pipeline.sha256_file(gate_path)
    assert result['source_inventories_before_after_unchanged'] is True
    assert result['research_api_requests_sent']==result['api_key_queries']==result['checkpoint_mutations']==0
    assert len(commit_calls)==2 and len(gate_calls)==3
    assert 'PHASE1_MANIFEST.json' not in result['products']
    for name,expected in result['products'].items():
        assert pipeline.sha256_file(out/name)==expected['sha256']
        assert (out/name).stat().st_size==expected['bytes']
    assert original_bytes=={str(p):p.read_bytes()for p in old.rglob('*')if p.is_file()}
