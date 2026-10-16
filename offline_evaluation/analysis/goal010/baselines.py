"""Frozen geometry-only baselines; construct every prediction before GT evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

from ...server_v2.core import Grid, Limits, ValidationError, evaluate_pair, execute, strict_json, validate_actions
from ...server_v2.alignment import AlignmentResourceLimit
from ...server_v2.data import load_scene, sha256_file
from ...server_v2.generation import object_hash, utc_now
from ..goal007 import statistics as old_stats
from ..goal007.integration import write_new_json
from ..goal008.pipeline import write_csv
from .factors import draw_counts
from .source_guard import require_gate, regular

STRUCTURES = ('filled_box', 'six_face_shell')
METRICS = ('absolute_occupancy_iou', 'aligned_occupancy_iou')
POLICIES = ('common_valid', 'known_failure0')
LABEL = 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC'


def box_actions(world_bbox, structure, limits=Limits()):
    """Only explicit bbox and fixed shape; no GT/material/count/style input."""
    if structure not in STRUCTURES or not isinstance(world_bbox, dict):
        raise ValidationError('fixed box structure and explicit bbox required')
    bounds = {}
    for axis in 'xyz':
        lo, hi = world_bbox.get(axis+'min'), world_bbox.get(axis+'max')
        if type(lo) is not int or type(hi) is not int or lo > hi:
            raise ValidationError('ordered inclusive integer bbox required')
        if abs(lo) > limits.max_coordinate_abs or abs(hi) > limits.max_coordinate_abs:
            raise ValidationError('bbox exceeds frozen coordinate limit')
        bounds[axis+'1'], bounds[axis+'2'] = lo, hi
    if structure == 'filled_box':
        actions = [{'op':'fill', **bounds, 'block':'stone'}]
    else:
        actions = []
        for axis in 'xyz':
            for end in ('1', '2'):
                face = dict(bounds); face[axis+'1'] = face[axis+'2'] = bounds[axis+end]
                actions.append({'op':'fill', **face, 'block':'stone'})
    return validate_actions(actions, limits)


def original_actions(dimensions_estimate, structure, limits=Limits()):
    """O input boundary: this function receives only estimated dimensions."""
    if not isinstance(dimensions_estimate, dict):
        raise ValidationError('O dimensions_estimate missing or not an object')
    dimensions = []
    for name in ('width', 'height', 'depth'):
        value = dimensions_estimate.get(name)
        if (type(value) not in (int, float, Decimal)
                or isinstance(value,float) and not math.isfinite(value)
                or isinstance(value,Decimal) and not value.is_finite() or value <= 0):
            raise ValidationError('O '+name+' must be a positive finite JSON number')
        if value > limits.max_coordinate_abs + 1:
            raise ValidationError('O dimension exceeds frozen coordinate limit')
        dimensions.append(max(1, round(value)))
    bbox = {axis+end: (0 if end=='min' else dimensions[i]-1)
            for i,axis in enumerate('xyz') for end in ('min','max')}
    return box_actions(bbox, structure, limits)


def partial_actions(world_bbox, structure, limits=Limits()):
    """P input boundary: only the bbox already disclosed in the fixed P input."""
    return box_actions(world_bbox, structure, limits)


def _records(package):
    """Bind immutable O/P inputs, without reading GT coordinates."""
    slots = old_stats.validate_rows(package['slots'], fixed_design=True)
    schedule = {r['schedule_index']:r for r in package['full_manifest']['identity']['full_schedule']}
    sources = {(r['scene_id'],r['description_provider']):r for r in package['descriptions']}
    records = {}
    for slot in slots:
        if slot['condition'] not in ('O','P'): continue
        record = schedule[slot['global_schedule_index']]
        key = (slot['scene_id'],slot['condition'],slot.get('description_model'))
        if (record['input_id'] != slot['input_id'] or record['input_sha256'] != slot['input_sha256']
                or hashlib.sha256(record['input_text'].encode()).hexdigest() != slot['input_sha256']):
            raise ValueError('baseline fixed input/selected slot binding mismatch')
        value = strict_json(record['input_text'])
        permitted = value.get('dimensions_estimate') if slot['condition']=='O' else value.get('world_bbox')
        result = {k:record[k] for k in ('scene_id','condition','description_model','input_id','input_sha256')}
        if slot['condition']=='O':
            source = sources[slot['scene_id'],slot['description_model']]
            path = regular(source['source_path'])
            if (source['input_id'] != record['input_id'] or source['input_sha256'] != record['input_sha256']
                    or sha256_file(path) != source['source_sha256'] or path.read_bytes().decode('utf-8') != record['input_text']):
                raise ValueError('O baseline input differs from exact saved description')
            result['source_sha256'] = source['source_sha256']
        result['permitted_input'] = permitted
        if key in records and records[key] != result:
            raise ValueError('two Builders do not share an identical baseline input')
        records[key] = result
    if len(records)!=240 or Counter(r['condition'] for r in records.values())!={'O':180,'P':60}:
        raise ValueError('baseline requires exactly180 O and60 P independent inputs')
    return [records[k] for k in sorted(records, key=lambda x:tuple(str(v) for v in x))]


def generate_predictions(records, destination, limits=Limits()):
    """No GT object/path is accepted; persistence precedes evaluation."""
    out = Path(destination); out.mkdir(parents=True, exist_ok=False)
    predictions = []
    for record in records:
        if record['condition'] not in ('O','P'):
            raise ValueError('only fixed O/P baseline conditions are supported')
        for structure in STRUCTURES:
            row = {k:v for k,v in record.items() if k!='permitted_input'}
            row.update(structure=structure, analysis_label=LABEL,
                prediction_id=object_hash({'input_id':record['input_id'],'structure':structure}),
                permitted_input=record['permitted_input'], status='PENDING', artificial_material='stone',
                material_metrics_used=False, artifacts={}, prediction_voxels=None)
            try:
                constructor = original_actions if record['condition']=='O' else partial_actions
                actions = constructor(record['permitted_input'], structure, limits)
                grid = execute(actions, limits)
            except ValidationError as exc:
                row.update(status='BASELINE_INPUT_OR_RESOURCE_REJECTED', reason=str(exc))
            else:
                prefix = out / row['prediction_id']
                actions_path = prefix.with_suffix('.actions.json')
                coordinates_path = prefix.with_suffix('.xyz.npy')
                write_new_json(actions_path, {'operations':actions})
                coordinates = np.asarray(sorted(grid.voxels), dtype=np.int32).reshape((-1,3))
                with coordinates_path.open('xb') as stream: np.save(stream, coordinates, allow_pickle=False)
                row.update(status='PREDICTION_SAVED', reason=None, prediction_voxels=len(grid.voxels),
                           prediction_bbox=grid.bbox(), prediction_dimensions_xyz=list(grid.dimensions()))
                row['artifacts'] = {name:{'path':p.name,'sha256':sha256_file(p),'bytes':p.stat().st_size}
                    for name,p in [('actions',actions_path),('coordinates',coordinates_path)]}
            predictions.append(row)
    manifest = {'status':'ALL_PREDICTIONS_FROZEN_BEFORE_GT_EVALUATION','at_utc':utc_now(),
        'predictions':predictions,'counts':dict(Counter(r['status'] for r in predictions)),
        'GT_voxels_read_during_construction':False,'Builder_replication_of_predictions':False,
        'coordinate_format':'NPY int32 Nx3 XYZ world; allow_pickle=False; constant stone',
        'research_api_requests_sent':0}
    write_new_json(out.parent/'baseline_predictions_manifest.json',manifest)
    return manifest


def _load_prediction(row, root, limits):
    artifacts = row['artifacts']
    for item in artifacts.values():
        name = Path(item['path'])
        if name.is_absolute() or '..' in name.parts: raise ValueError('prediction path traversal')
        path = regular(Path(root)/name)
        if sha256_file(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:
            raise ValueError('prediction artifact changed')
    actions = strict_json((Path(root)/artifacts['actions']['path']).read_bytes().decode('utf-8'),limits)['operations']
    prediction = execute(actions,limits)
    coordinates = np.load(Path(root)/artifacts['coordinates']['path'],allow_pickle=False)
    expected = np.asarray(sorted(prediction.voxels),dtype=np.int32).reshape((-1,3))
    if coordinates.dtype!=np.int32 or coordinates.shape!=expected.shape or not np.array_equal(coordinates,expected):
        raise ValueError('persisted coordinate/actions mismatch')
    if len(prediction.voxels)!=row['prediction_voxels']: raise ValueError('prediction count changed')
    return prediction


def evaluate_predictions(predictions, package, root, limits):
    from .independent_geometry import check_box, IndependentGeometryResourceLimit
    sources = {r['scene_id']:r for r in package['descriptions']}
    strata = {r['scene_id']:r['size_rank_stratum'] for r in package['cohort']['scenes']}
    rows = []; current_scene = None
    for index, saved in enumerate(predictions):
        row = dict(saved)
        row.update(absolute_occupancy_iou=None,aligned_occupancy_iou=None,independent_check_status='NOT_RUN')
        if current_scene != row['scene_id']:
            source = sources[row['scene_id']]; current_scene = row['scene_id']
            scene = load_scene(source['gt_descriptor']['building_directory'],limits)
            if scene.scene_id!=current_scene or object_hash(scene.grid.to_dict())!=source['gt_descriptor']['grid_sha256']:
                raise ValueError('GT axes/origin/content hash mismatch')
        gt = scene.grid.as_world()
        storage_volume = math.prod(scene.bbox[a+'max']-scene.bbox[a+'min']+1 for a in 'xyz')
        row.update(gt_bbox_sha256=source['gt_descriptor']['bbox_sha256'],gt_voxels_sha256=source['gt_descriptor']['voxels_sha256'],
            gt_grid_sha256=source['gt_descriptor']['grid_sha256'], gt_occupied_count=len(gt.voxels),
            gt_storage_bbox=scene.bbox,gt_occupied_bbox=gt.bbox(),gt_storage_bbox_volume=storage_volume,
            gt_storage_density=len(gt.voxels)/storage_volume,size_rank_stratum=strata[current_scene])
        if row['status']=='PREDICTION_SAVED':
            prediction = _load_prediction(row,root,limits)
            try:
                actual = evaluate_pair(gt,prediction,limits=limits,alignment_mode='strict_translation')
                independent = check_box(gt,prediction,row['structure'],limits)
            except (AlignmentResourceLimit, IndependentGeometryResourceLimit) as exc:
                row.update(status=exc.metadata['status'],reason=str(exc),alignment_failure=exc.metadata)
            else:
                for frame in ('absolute','aligned'):
                    if (actual[frame]['occupancy_intersection']!=independent[frame+'_intersection'] or
                            abs(actual[frame]['occupancy_iou']-independent[frame+'_occupancy_iou'])>1e-12):
                        raise ValueError('independent baseline occupancy disagreement')
                if actual['alignment']['shift_xyz']!=independent['shift_xyz']:
                    raise ValueError('independent baseline alignment tie disagreement')
                row.update(status='success',metric_version=actual['metric_version'],alignment=actual['alignment'],
                    independent_check=independent,independent_check_status='PASS',
                    **{frame+'_occupancy_iou':actual[frame]['occupancy_iou'] for frame in ('absolute','aligned')})
                if row['condition']=='P' and row['structure']=='filled_box':
                    box = row['permitted_input']
                    if not all(all(box[a+'min']<=c[i]<=box[a+'max'] for i,a in enumerate('xyz')) for c in gt.voxels):
                        raise ValueError('P disclosed bbox does not contain all GT occupied coordinates')
                    box_volume = math.prod(box[a+'max']-box[a+'min']+1 for a in 'xyz')
                    if row['absolute_occupancy_iou']!=len(gt.voxels)/box_volume:
                        raise ValueError('P filled-box independent density identity failed')
                    row.update(P_bbox_volume=box_volume,P_density_identity=len(gt.voxels)/box_volume,
                               P_storage_volume_difference=storage_volume-box_volume,P_density_identity_status='PASS')
        rows.append(row)
        if (index+1)%8==0: print(f'baseline evaluated {index+1}/{len(predictions)}',flush=True)
    return rows


def building_interval(values, frame):
    """Equal-building mean with all planned clusters retained in resampling."""
    if len(set(frame))!=len(frame) or not set(values)<=set(frame): raise ValueError('invalid bootstrap frame')
    array = np.array([values.get(s,np.nan) for s in frame],dtype=float)
    finite = np.isfinite(array)
    if any(not math.isfinite(v) for v in values.values()): raise ValueError('nonfinite included building')
    result = {'mean_delta':float(np.mean(array[finite])) if finite.any() else None,'ci95':None,
        'independent_buildings':int(finite.sum()),'planned_buildings':len(frame),'bootstrap_planned':2000,
        'bootstrap_valid':0,'status':'INSUFFICIENT_BUILDINGS'}
    if finite.sum()<2: return result
    counts = draw_counts(len(frame)); denominator = counts[:,finite].sum(axis=1)
    selected = denominator>0
    samples = (counts[:,finite]*array[finite]).sum(axis=1)[selected]/denominator[selected]
    result.update(bootstrap_valid=len(samples),status='UNSTABLE_BOOTSTRAP')
    if len(samples)>=1900:
        result.update(ci95=np.quantile(samples,[.025,.975],method='linear').tolist(),status='ESTIMABLE')
    return result


def compare_baselines(baselines, slots, *, fixed_design=True):
    slots = old_stats.validate_rows(slots,fixed_design=fixed_design)
    slot_index = {(r['scene_id'],r['condition'],r.get('description_model'),r['provider']):r for r in slots}
    frame = sorted({r['scene_id'] for r in slots})
    if len({r['prediction_id'] for r in baselines})!=len(baselines): raise ValueError('duplicate baseline prediction')
    pairs = []
    for row in baselines:
        for builder in ('openai','anthropic'):
            slot = slot_index[row['scene_id'],row['condition'],row['description_model'],builder]
            if slot['input_id']!=row['input_id'] or slot['input_sha256']!=row['input_sha256']:
                raise ValueError('baseline/Builder input mismatch')
            for metric in METRICS:
                observed = old_stats.legacy._metric(slot,metric)
                for policy in POLICIES:
                    value = observed
                    imputed = False
                    if value is None and policy=='known_failure0' and old_stats.legacy.attempted_failure(slot) and slot['usage_known']:
                        value = 0.; imputed=True
                    baseline = row[metric] if row['status']=='success' else None
                    reason = 'baseline_'+row['status'] if baseline is None else 'builder_'+slot['status'] if value is None else None
                    pairs.append({**{k:row[k] for k in ('prediction_id','scene_id','condition','description_model','structure','input_id','input_sha256')},
                        'builder':builder,'metric':metric,'policy':policy,'global_schedule_index':slot['global_schedule_index'],
                        'attempt_id':slot['attempt_id'],'source_label':slot.get('source_label'),
                        'source_plan_identity':slot.get('source_plan_identity'),'builder_status':slot['status'],
                        'builder_failure_kind':slot.get('failure_kind'),'baseline_status':row['status'],
                        'failure0_imputed':imputed,'exclusion_reason':reason,'builder_value':value,'baseline_value':baseline,
                        'delta':value-baseline if reason is None else None})
    records = []
    for condition in ('O','P'):
        for provider in (None,'openai','anthropic','gemini') if condition=='O' else (None,):
            for structure in STRUCTURES:
                for builder in ('openai','anthropic'):
                    for metric in METRICS:
                        for policy in POLICIES:
                            selected = [r for r in pairs if r['condition']==condition and r['structure']==structure and r['builder']==builder
                                and r['metric']==metric and r['policy']==policy and (provider is None or r['description_model']==provider)]
                            grouped = defaultdict(list)
                            for r in selected:
                                if r['delta'] is not None: grouped[r['scene_id']].append(r)
                            differences = {s:sum(r['delta'] for r in rs)/len(rs) for s,rs in grouped.items()}
                            means = {kind:float(np.mean([np.mean([r[kind] for r in rs]) for rs in grouped.values()])) if grouped else None
                                     for kind in ('builder_value','baseline_value')}
                            records.append({'analysis_label':LABEL,'condition':condition,'description_provider':provider or 'all',
                                'structure':structure,'builder':builder,'metric':metric,'policy':policy,
                                'difference_direction':'Builder minus baseline','estimator':'equal-building mean of within-building eligible input differences',
                                'planned_pairs':len(selected),'common_pairs':sum(len(v) for v in grouped.values()),
                                'excluded_pairs':sum(r['delta'] is None for r in selected),
                                'failure0_imputed_pairs':sum(r['failure0_imputed'] and r['delta'] is not None for r in selected),
                                'exclusion_counts':dict(Counter(r['exclusion_reason'] for r in selected if r['exclusion_reason'])),
                                **means,**building_interval(differences,frame),'p_values_computed':False})
    if len(records)!=80: raise AssertionError('fixed80 baseline contrasts required')
    return records,pairs


def size_distributions(rows):
    result = []
    for condition in ('O','P'):
        for structure in STRUCTURES:
            for stratum in (None,0,1,2,3,4):
                selected = [r for r in rows if r['condition']==condition and r['structure']==structure
                    and (stratum is None or r['size_rank_stratum']==stratum)]
                for metric in METRICS:
                    by_scene = defaultdict(list)
                    for row in selected:
                        if row['status']=='success': by_scene[row['scene_id']].append(row[metric])
                    values = [float(np.mean(v)) for v in by_scene.values()]
                    result.append({'condition':condition,'structure':structure,'size_rank_stratum':'all' if stratum is None else stratum,
                        'metric':metric,'planned_inputs':len(selected),'eligible_inputs':sum(map(len,by_scene.values())),
                        'independent_buildings':len(values),'minimum':min(values) if values else None,'maximum':max(values) if values else None,
                        'q1_median_q3':np.quantile(values,[.25,.5,.75],method='linear').tolist() if values else None,
                        'estimator':'distribution of equal-building within-provider-averaged baseline IoU'})
    return result


def run_baselines(package,destination,*,gate,specification_sha256):
    started = time.monotonic(); require_gate(gate,specification_sha256)
    out = Path(destination).absolute()
    if out.exists() or not out.is_relative_to(Path('/root/minecraft-runs')):
        raise ValueError('new research run output required')
    out.mkdir(parents=True,exist_ok=False)
    limits = Limits(**package['cohort']['executor_limits'])
    records = _records(package)
    manifest = generate_predictions(records,out/'predictions',limits)
    print('all480 predictions frozen; GT evaluation begins',flush=True)
    rows = evaluate_predictions(manifest['predictions'],package,out/'predictions',limits)
    comparisons,pairs = compare_baselines(rows,package['slots'])
    sizes = size_distributions(rows)
    write_new_json(out/'baseline_per_building.json',rows)
    write_csv(out/'baseline_per_building.csv',rows)
    write_new_json(out/'baseline_matched_deltas.json',comparisons)
    write_csv(out/'baseline_matched_deltas.csv',comparisons)
    write_csv(out/'baseline_pairs.csv',pairs)
    write_csv(out/'baseline_size_distributions.csv',sizes)
    summary = {'status':'BASELINE_PHASE_COMPLETED','analysis_label':LABEL,'gate':gate,
        'prediction_count':len(rows),'independent_buildings':len({r['scene_id'] for r in rows}),
        'status_counts':dict(Counter(r['status'] for r in rows)),'comparisons':len(comparisons),'expanded_pairs':len(pairs),
        'independent_verified_predictions':sum(r['independent_check_status']=='PASS' for r in rows),
        'P_density_checks':sum(r.get('P_density_identity_status')=='PASS' for r in rows),'material_metrics_used':False,
        'research_api_requests_sent':0,'performance':{'seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    summary['products'] = {p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    write_new_json(out/'BASELINE_MANIFEST.json',summary)
    return summary
