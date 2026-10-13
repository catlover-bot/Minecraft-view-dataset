"""Representative CPU checks using the downloaded/restored Goal002 code only."""
from pathlib import Path
import json
import sys
import time
import resource

project, run, output = map(Path, sys.argv[1:4])
sys.path.insert(0, str(project))
from offline_evaluation.server_v2.io_utils import ROOT, atomic_json, read_json, sha256
from offline_evaluation.server_v2.cohort_v4 import build_v4_cohort, build_inputs
from offline_evaluation.server_v2.data import load_scene
from offline_evaluation.server_v2.core import execute, evaluate_pair
from offline_evaluation.server_v2.oracle import encode_exact_spec, decode_exact_spec, gt_to_actions
from offline_evaluation.server_v2.geometry_data import load_geometry, storage_truncation_evidence
from offline_evaluation.server_v2.generation import dry_run_generation, evaluate_generation, _load_run
from offline_evaluation.server_v2.smoke import dry_run_smoke_campaign
from offline_evaluation.server_v2.analysis_v4 import collect_planned_outcomes, summarize_v4

assert ROOT.resolve() == project.resolve(), 'Restored code must resolve from the new project'
started = time.monotonic()
saved = read_json(run / 'cohort.v4.json')
rebuilt = build_v4_cohort(budgets=saved['input_budgets'])
assert rebuilt['cohort_sha256'] == saved['cohort_sha256'] == 'eeed4b62047106f4ce75d853c4e33dc22cac812224282ceca9d6b84f6b696f63'
for split in ('evaluation', 'initial_smoke', 'smoke'):
    assert build_inputs(rebuilt, split) == read_json(run / f'inputs.{split}.json')
controls = []
for building in ('building_060', 'building_075'):
    scene = load_scene(project / 'datasets/buildings_100_v4' / building)
    before = sha256(scene.paths['voxels'])
    assert scene.grid.voxels
    assert decode_exact_spec(encode_exact_spec(scene.grid)).voxels == scene.grid.voxels
    assert execute(gt_to_actions(scene.grid)).voxels == scene.grid.voxels
    tests = []
    for displacement in ((1,-1,1), (20,-3,-17), (-100,7,80)):
        result = evaluate_pair(scene.grid,scene.grid.shifted(displacement),alignment_mode='strict_translation')
        assert result['alignment']['search_complete'] and result['aligned']['occupancy_iou'] == 1
        assert result['alignment']['shift_xyz'] == [-v for v in displacement]
        tests.append(result['alignment']['shift_xyz'])
    result = evaluate_pair(scene.grid.shifted((50000,-20000,70000)),
                           scene.grid.shifted((50020,-20003,69983)), alignment_mode='strict_translation')
    assert result['aligned']['occupancy_iou'] == 1 and result['alignment']['shift_xyz'] == [-20,3,17]
    assert sha256(scene.paths['voxels']) == before
    controls.append({'scene_id':scene.scene_id,'nonempty':True,'exact_roundtrip':True,
                     'executor_control':True,'known_translation_checks':len(tests),'far_origin':True})

diagnostics = read_json(project / 'reports/server_only/goal002/material_diagnostics.json')
selected = [diagnostics[0]]
selected += [row for row in diagnostics if row['confirmed_storage_truncation']][:2]
selected += [row for row in diagnostics if any('_or_' in entry['stored_label'] for entry in row['invalid_labels'])][:2]
seen, material_checks = set(), []
for row in selected:
    source = next(s for s in row['original_sources'] if s['path'].endswith('/voxels.npy'))
    if source['path'] in seen:continue
    seen.add(source['path'])
    path=project/source['path']
    before=sha256(path)
    assert before==source['sha256']
    read=load_geometry(path,path.parent/'bbox.json')
    evidence=storage_truncation_evidence(read,read_json(path.parent/'actions.json'))
    assert read.geometry_valid and not read.material_valid and read.material_grid is None
    assert sha256(path)==before
    material_checks.append({'source_path':source['path'],'geometry_valid':True,'material_valid':False,
                            'confirmed_truncation':bool(evidence),'original_sha256':before})

dry=[]
directories=sorted(run.glob('builder_*'))
for directory in directories:
    manifest,config,records,prompt,state=_load_run(directory)
    assert not config.api_allowed and not state['attempts']
    summary=dry_run_generation(directory)
    rows=evaluate_generation(directory,{},alignment_radius=1,alignment_mode='strict_translation')
    assert all(r['status']=='NOT_RUN' and r['metrics'] is None and not r['attempted'] for r in rows)
    dry.append({'directory':directory.name,'inputs':len(records),'api_allowed':False,'attempts':0})
smoke_dry=dry_run_smoke_campaign(run/'smoke_proposal')
assert not smoke_dry['api_allowed'] and smoke_dry['target_count']==16
counts={}
for split,count in [('evaluation',960),('initial_smoke',16)]:
    planned=collect_planned_outcomes([d for d in directories if read_json(d/'selection.json')['split']==split])
    assert len(planned)==count and all(r['status']=='NOT_RUN' and r['metrics'] is None for r in planned)
    summary=summarize_v4(planned,smoke=split=='initial_smoke')
    assert sum(g['planned_rows'] for g in summary['groups'])==count
    if split=='initial_smoke':assert summary['families']==[]
    counts[split]=count
assert sum(len(read_json(p)['attempts']) for p in run.rglob('state.json'))==0
result={'status':'PASS','restored_project':str(ROOT),'scope':'Full v4 source/cohort/input reconstruction and representative CPU execution; not a second all200/all1200 CPU rerun',
        'cohort_sha256':rebuilt['cohort_sha256'],'cohort_population_verified':100,
        'exact_inputs_equal':{'evaluation':480,'initial_smoke':8,'smoke':80},
        'representative_gt_controls':controls,'legacy_material_checks':material_checks,
        'disabled_builder_dry_runs':dry,'preserved_planned_summary_counts':counts,
        'api_key_present':smoke_dry['api_key_present'],'research_api_requests_sent':0,
        'wall_seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
atomic_json(output,result)
print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))
