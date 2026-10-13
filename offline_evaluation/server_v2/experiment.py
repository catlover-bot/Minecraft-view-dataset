"""New cohort preparation and CPU verification, separated from paid generation."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import subprocess
import time
from .io_utils import ROOT, atomic_json, read_json, sha256, writable

SEED = 20261013

def _scenes():
    return sorted(path for version in ('v1', 'v4')
                  for path in (ROOT / f'datasets/buildings_100_{version}').glob('building_*')
                  if (path / 'gt/voxels.npy').is_file())

def _description_paths(scene):
    paths = list(scene.glob('description_*/description.json'))
    paths += list((ROOT / 'outputs/i2t2b' / scene.parent.name / scene.name).glob('description_*/description.json'))
    by_model = {}
    for path in sorted(paths):
        folder = path.parent.name
        model = next((m for m in ('openai', 'anthropic', 'gemini') if m in folder), None)
        if model:
            if model in by_model and sha256(path) != sha256(by_model[model]):
                raise ValueError('Ambiguous multiple descriptions for scene/provider; select explicitly')
            by_model[model] = path
    return by_model

def prepare(output_dir):
    """Freeze size/version strata before examining prediction scores."""
    from .data import load_scene
    output_dir = writable(output_dir)
    marker = output_dir / 'subset.json'
    if marker.exists():
        raise FileExistsError('Subset is immutable; reuse or choose a new cohort directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows, selections = [], []
    for version in ('v1', 'v4'):
        scenes = [load_scene(p) for p in _scenes() if p.parent.name.endswith(version)]
        scenes.sort(key=lambda s: (len(s.grid.voxels), s.scene_id))
        if len(scenes) < 70:
            raise ValueError('Insufficient GT scenes for planned version/size stratification')
        for stratum in range(5):
            group = scenes[len(scenes)*stratum//5:len(scenes)*(stratum+1)//5]
            order = rng.sample(group, len(group))
            for index, scene in enumerate(order[:7]):
                split = 'smoke' if index == 0 else 'evaluation'
                selections.append({'scene_id': scene.scene_id, 'dataset_version': version,
                                   'gt_occupied': len(scene.grid.voxels), 'size_rank_stratum': stratum,
                                   'split': split, 'scene_path': str(Path(scene.paths['voxels']).parent.parent.relative_to(ROOT))})
    assert len({r['scene_id'] for r in selections}) == 70
    subset = {'provenance': 'new_implementation', 'cohort_id': 'server_v2_saved_description_subset_20261013',
              'seed': SEED, 'selection': 'within-version occupancy rank quintiles, 1 smoke + 6 evaluation per stratum',
              'freeze_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'selection_uses_prediction_results': False, 'unseen_data': False, 'scenes': selections}
    atomic_json(marker, subset)
    for selection in selections:
        scene_path = ROOT / selection['scene_path']
        descriptions = _description_paths(scene_path)
        for model in ('openai', 'anthropic', 'gemini'):
            if model not in descriptions:
                rows.append(dict(selection, description_model=model, condition='Original', status='MISSING', input_text=None))
                rows.append(dict(selection, description_model=model, condition='Canonicalized', status='MISSING', input_text=None))
                continue
            path = descriptions[model]
            original = path.read_text()
            parsed = json.loads(original)
            # Lossless key-order/whitespace canonicalization only, using saved text.
            canonical = json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
            for condition, text in (('Original', original), ('Canonicalized', canonical)):
                rows.append(dict(selection, description_model=model, condition=condition, status='prepared',
                                 input_text=text, source_path=path.relative_to(ROOT).as_posix(),
                                 source_sha256=sha256(path), input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                 transformation='identity' if condition == 'Original' else 'new_lossless_json_canonicalization_v1',
                                 historical_canonicalized_replay='BLOCKED_MISSING_ORIGINALS', oracle=False))
    atomic_json(output_dir / 'inputs.json', rows)
    counts = {split: {condition: sum(r['split'] == split and r['condition'] == condition and r['status'] == 'prepared' for r in rows)
                      for condition in ('Original', 'Canonicalized')} for split in ('smoke', 'evaluation')}
    atomic_json(output_dir / 'preparation.json', {'counts': counts, 'missing_rows': sum(r['status']=='MISSING' for r in rows),
                 'api_status': 'NOT_RUN', 'requests_sent': 0, 'cohort_id': subset['cohort_id'],
                 'alternative_plan': 'Evaluate available saved descriptions; acquire v1 Gemini descriptions in a separately authorized new cohort to reach 360. Never invent historical inputs.'})
    return counts

def _verify_one(path, output_dir):
    from .core import execute, evaluate_pair
    from .data import load_scene
    from .oracle import encode_exact_spec, decode_exact_spec, gt_to_actions, partial_oracle
    scene = load_scene(path)
    spec = encode_exact_spec(scene.grid)
    decoded = decode_exact_spec(spec)
    actions = gt_to_actions(scene.grid)
    controlled = execute(actions)
    shift_test = evaluate_pair(scene.grid, scene.grid.shifted((1,-1,1)), alignment_radius=1)
    exact = decoded.voxels == scene.grid.voxels
    control = controlled.voxels == scene.grid.voxels
    recovered = shift_test['aligned']['occupancy_iou'] == 1.0
    directory = output_dir / 'oracle' / scene.scene_id
    atomic_json(directory / 'exact_spec.json', spec)
    atomic_json(directory / 'control_actions.json', actions)
    atomic_json(directory / 'partial_spec.json', partial_oracle(scene.grid))
    record = {'scene_id': scene.scene_id, 'provenance': 'new_implementation', 'gt_occupied': len(scene.grid.voxels),
              'exact_roundtrip': exact, 'executor_control': control, 'known_translation_recovered': recovered,
              'exact_spec_bytes': (directory/'exact_spec.json').stat().st_size, 'actions': len(actions),
              'spec_sha256': sha256(directory/'exact_spec.json'), 'gt_sha256': sha256(scene.paths['voxels']),
              'shift_test': shift_test, 'llm_builder_success': None, 'api_status': 'NOT_RUN'}
    atomic_json(directory / 'result.json', record)
    return record

def verify_gt(output_dir, workers=2):
    output_dir = writable(output_dir)
    if (output_dir/'gt_summary.json').exists():
        raise FileExistsError('Completed GT verification is immutable; choose a new run directory')
    if not 1 <= workers <= 2:
        raise ValueError('Initial worker limit is 2')
    if shutil.disk_usage(ROOT).free < 10*1024**3:
        raise ValueError('Less than 10GiB free')
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda path: _verify_one(path, output_dir), _scenes()))
    summary = {'provenance': 'new_implementation', 'gt_scenes': len(rows),
               'oracle_roundtrip_pass': sum(r['exact_roundtrip'] for r in rows),
               'executor_control_pass': sum(r['executor_control'] for r in rows),
               'known_translation_pass': sum(r['known_translation_recovered'] for r in rows),
               'wall_seconds': time.monotonic()-start, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'disk_free_bytes': shutil.disk_usage(ROOT).free, 'workers': workers,
               'alignment_search': 'exact exhaustive translation within [-1,1]^3; no rotation or scale',
               'requests_sent': 0, 'api_status': 'NOT_RUN', 'llm_builder_success_rate': None}
    atomic_json(output_dir / 'gt_results.json', rows)
    atomic_json(output_dir / 'gt_summary.json', summary)
    return summary

def oracle_inputs(output_dir):
    output_dir = writable(output_dir)
    subset = read_json(output_dir/'subset.json')
    rows = []
    for scene in subset['scenes']:
        for condition, name in (('Partial Oracle','partial_spec.json'),('Exact-spec Oracle','exact_spec.json')):
            path = output_dir/'oracle'/scene['scene_id']/name
            text = path.read_text()
            rows.append(dict(scene, condition=condition, description_model=None, status='prepared', oracle=True,
                             input_text=text, input_sha256=sha256(path), source_path=f"oracle/{scene['scene_id']}/{name}",
                             diagnostic_only=True))
    atomic_json(output_dir/'oracle_inputs.json', rows)
    return {'smoke_per_oracle_condition': 10, 'evaluation_per_oracle_condition': 60,
            'replicated_per_description_model': False, 'executor_control': 'deterministic, no LLM; all200 verified separately'}

def reevaluate(output_dir, alignment_radius=1, workers=2):
    from .core import evaluate_pair
    from .data import load_scene, load_grid
    output_dir = writable(output_dir)
    if (output_dir/'reevaluation_summary.json').exists():
        raise FileExistsError('Completed reevaluation is immutable; choose a new run directory')
    if not 1 <= workers <= 2:
        raise ValueError('Initial worker limit is 2')
    if shutil.disk_usage(ROOT).free < 10*1024**3:
        raise ValueError('Less than 10GiB free')
    paths = sorted((ROOT/'outputs/i2t2b').glob('*/building_*/rebuild_world_*/voxels.npy'))
    start=time.monotonic()
    def one(path):
        scene_path=ROOT/'datasets'/path.parents[2].name/path.parents[1].name
        scene=load_scene(scene_path)
        row={'scene_id':scene.scene_id,'condition':path.parent.name,'prediction_path':path.relative_to(ROOT).as_posix(),
             'prediction_sha256':sha256(path),'provenance':'new_implementation_metrics_on_saved_i2t2b_v5_output',
             'historical_arr_replay':False}
        try:
            pred=load_grid(path,path.parent/'bbox.json')
            row.update(status='success',metrics=evaluate_pair(scene.grid,pred,alignment_radius=alignment_radius))
        except (ValueError, OSError) as exc:
            row.update(status='execution_failure',metrics=None,error_type=type(exc).__name__)
        atomic_json(output_dir/'reevaluation'/scene.scene_id/(path.parent.name+'.json'),row)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(one,paths))
    atomic_json(output_dir/'reevaluation_results.json',rows)
    summary={'count':len(rows),'success':sum(r['status']=='success' for r in rows),
             'failure':sum(r['status']!='success' for r in rows),'alignment_radius':alignment_radius,
             'wall_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'provenance':'new_implementation_metrics_on_saved_i2t2b_v5_output',
             'historical_arr_replay':'BLOCKED_MISSING_ORIGINALS','requests_sent':0}
    atomic_json(output_dir/'reevaluation_summary.json',summary)
    return summary

def summarize(output_dir):
    from .statistics import paired_cluster_ci
    output_dir=writable(output_dir)
    rows=read_json(output_dir/'reevaluation_results.json') if (output_dir/'reevaluation_results.json').exists() else []
    conditions=sorted({r['condition'] for r in rows})
    groups=[]
    for condition in conditions:
        successful=[r for r in rows if r['condition']==condition and r['status']=='success']
        groups.append({'condition':condition,'success':len(successful),
                       'failures':sum(r['condition']==condition and r['status']!='success' for r in rows),
                       'absolute_iou_mean':sum(r['metrics']['absolute']['occupancy_iou'] for r in successful)/len(successful) if successful else None,
                       'aligned_iou_mean':sum(r['metrics']['aligned']['occupancy_iou'] for r in successful)/len(successful) if successful else None})
    comparisons=[]
    for condition in conditions:
        paired=[{'scene_id':r['scene_id'],'status_a':r['status'],'status_b':r['status'],
                 'a':r['metrics']['absolute']['occupancy_iou'] if r['metrics'] else None,
                 'b':r['metrics']['aligned']['occupancy_iou'] if r['metrics'] else None}
                for r in rows if r['condition']==condition]
        comparisons.append(dict(condition=condition,**paired_cluster_ci(paired,'aligned_minus_absolute_occupancy_iou')))
    builder_summaries=[]
    for directory in sorted(output_dir.glob('builder_*')):
        evaluation=directory/'evaluation.json'
        if not evaluation.is_file():continue
        generated=read_json(evaluation)['rows']
        states={}
        pairs={}
        for row in generated:
            key=(row['condition'],row['status'])
            states[key]=states.get(key,0)+1
            if row['condition'] in ('Original','Canonicalized'):
                pairs.setdefault((row['scene_id'],row.get('description_model')), {})[row['condition']]=row
        deltas=[]
        for (scene,model),pair in sorted(pairs.items()):
            a,b=pair.get('Original'),pair.get('Canonicalized')
            if not a or not b:continue
            deltas.append({'scene_id':scene,'a':a['metrics']['absolute']['occupancy_iou'] if a['metrics'] else None,
                           'b':b['metrics']['absolute']['occupancy_iou'] if b['metrics'] else None,
                           'status_a':a['status'],'status_b':b['status']})
        builder_summaries.append({'directory':directory.name,'conditions':[{'condition':c,'status':s,'rows':n} for (c,s),n in sorted(states.items())],
                                  'original_vs_canonicalized':paired_cluster_ci(deltas,'canonicalized_minus_original_absolute_iou')})
    result={'groups':groups,'paired_building_cluster_ci':comparisons,'builder_summaries':builder_summaries,
            'multiple_comparisons':'Descriptive CIs only; no significance claims or unadjusted p-values. Holm utility tested for future registered hypothesis family.',
            'oracle_difference_interpretation':'Diagnostic condition differences; not pure language information loss',
            'research_api':'NOT_RUN','requests_sent':0}
    atomic_json(output_dir/'summary.json',result)
    return {'groups':len(groups),'paired_cluster_ci':len(comparisons),'requests_sent':0}
