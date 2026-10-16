"""Ordered CPU-only addendum; retain every fixed slot and every frozen baseline."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import shutil
import time

from ..goal010 import source_guard as guard, pipeline as oldpipe, exact_oracle, baselines
from ...server_v2.core import Limits, payload_to_grid, execute, validate_actions
from ...server_v2.data import load_scene
from ...server_v2.generation import object_hash
from ...server_v2.analysis_v4 import attempted_failure
from . import metrics, independent, statistics

ROOT=guard.ROOT
REPORT=ROOT/'reports/server_only/goal010v2'
OLD=ROOT/'reports/server_only/goal010'
OLD_MANIFEST_SHA='dcad6f41f991425063c239798fec9b8d3bec2c0474fde5f901205a67d1432d7f'
SOURCE_SHA='01fcabeac3c37fd39868d90234781773ddf6215cca1f1af5fb331111139f9b51'
CODE=[f'offline_evaluation/analysis/goal010v2/{n}.py' for n in ('__init__','metrics','independent','pipeline','statistics')]
SPECS=[f'reports/server_only/goal010v2/{n}' for n in ('posthoc_specification.md','METRIC_SPEC.md','METRIC_REGISTRY.json','STATS_SPEC.md')]
STAGES=('source','pilot1','pilot10','full','summary')
COORDINATES='XYZ_WORLD_Y_UP_INTEGER'
PROTOCOL='server_v2.2_strict_translation'
ZERO={'research_api_requests_sent':0,'model_queries':0,'token_count_api_requests':0,'api_key_queries':0,
      'human_experiments':0,'checkpoint_mutations':0,'draft_updates':0}
write_json=oldpipe.write_json
write_csv=oldpipe.write_csv
sha=guard.sha256_file


def compact_json(path,value):
    with Path(path).open('x') as f: json.dump(value,f,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False); f.write('\n')


def verify_old():
    if sha(guard.regular(OLD/'MANIFEST.json'))!=OLD_MANIFEST_SHA: raise ValueError('old Goal010 manifest changed')
    m=guard.read_json(OLD/'MANIFEST.json'); guard.verify_products(OLD,m['products']); guard.verify_products(ROOT,m['code_files'])
    copy=guard.read_json(OLD/'analysis/PUBLIC_COPY_RECEIPT.json'); run=Path(copy['run_root'])
    for phase in range(6):
        name=f'PHASE{phase}_RECEIPT.json'
        if sha(guard.regular(run/name))!=sha(OLD/'analysis'/name): raise ValueError('old phase receipt copy differs')
        receipt=guard.read_json(run/name); guard.verify_products(run,receipt['products'])
    oldpipe.verify_predictions(run)
    return run


def binding(code_commit,spec_commit):
    present={str(p.relative_to(ROOT)) for p in (ROOT/'offline_evaluation/analysis/goal010v2').glob('*.py')}
    if present!=set(CODE): raise ValueError('production code allowlist mismatch')
    code=guard.verify_commit(code_commit,CODE+SPECS); specs=guard.verify_commit(spec_commit,SPECS)
    if any(code[n]!=specs[n] for n in SPECS): raise ValueError('spec changed after freeze')
    registry=guard.read_json(REPORT/'METRIC_REGISTRY.json')
    for entry in registry['metrics']:
        if entry['implementation_sha256']!=code['offline_evaluation/analysis/goal010v2/metrics.py']:
            raise ValueError('metric registry implementation hash mismatch')
    return {'code_commit':code_commit,'specification_commit':spec_commit,'hashes':code,
            'specification_sha256':specs['reports/server_only/goal010v2/METRIC_SPEC.md'],
            'old_goal010_manifest_sha256':OLD_MANIFEST_SHA}


def safe_root(value,stage):
    import re
    path=Path(value)
    if not path.is_absolute() or path.parent!=Path('/root/minecraft-runs') or not re.fullmatch(r'goal010-v2-\d{8}T\d{6}Z(?:-[a-z0-9-]+)?',path.name):
        raise ValueError('fresh goal010-v2 UTC run path required')
    if any(p.is_symlink() for p in (path,*path.parents)): raise ValueError('symlink run rejected')
    if stage==0 and path.exists(): raise FileExistsError('existing run must never be overwritten')
    if stage and not path.is_dir(): raise ValueError('prior source stage required')
    if shutil.disk_usage(path.parent).free<10*1024**3: raise ValueError('at least10 GiB free required')
    return path


def previous(root,stage,bound):
    result={}
    for i in range(stage):
        name=f'{i}_{STAGES[i]}_RECEIPT.json'; receipt=guard.read_json(root/name)
        if (receipt['status']!='PASS' or receipt.get('stage')!=i or receipt['binding']!=bound
                or receipt['source_package_sha256']!=SOURCE_SHA):
            raise ValueError('previous stage mismatch or incomplete')
        if i in (1,2,3):
            count={1:1,2:10,3:60}[i]
            stage_result=receipt.get('result',{})
            if stage_result.get('buildings')!=count or stage_result.get('slots')!=count*16 or stage_result.get('baseline_predictions')!=count*8:
                raise ValueError('previous pilot/full scope mismatch')
        guard.verify_products(root,receipt['products']); result[name]=sha(root/name)
    return result


class SampleResourceLimit(RuntimeError): pass


@contextmanager
def sample_budget(cpu_seconds=180,wall_seconds=240):
    def stop(signum,frame): raise SampleResourceLimit(f'CPU{cpu_seconds}_SECONDS_OR_WALL{wall_seconds}_SECONDS')
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGPROF)}
    previous_timers={t:signal.getitimer(t) for t in (signal.ITIMER_REAL,signal.ITIMER_PROF)}
    started=time.monotonic(); cpu_started=time.process_time()
    for s in handlers: signal.signal(s,stop)
    for timer,limit in ((signal.ITIMER_REAL,wall_seconds),(signal.ITIMER_PROF,cpu_seconds)):
        prior=previous_timers[timer][0]
        signal.setitimer(timer,min(limit,prior) if prior>0 else limit)
    try: yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.setitimer(signal.ITIMER_PROF,0)
        for s,h in handlers.items(): signal.signal(s,h)
        for timer,(remaining,interval) in previous_timers.items():
            if remaining>0:
                elapsed=time.monotonic()-started if timer==signal.ITIMER_REAL else time.process_time()-cpu_started
                signal.setitimer(timer,max(.001,remaining-elapsed),interval)


def null_metrics(reason):
    registry=guard.read_json(REPORT/'METRIC_REGISTRY.json')
    ids=[r['metric_id'] for r in registry['metrics'] if r['section']=='frame']
    return {f:{i:None for i in ids} for f in ('absolute','aligned')}, {f:{i:reason for i in ids} for f in ('absolute','aligned')}


def checked_artifact(root,record):
    p=guard.safe_child(root,record['path'])
    if sha(p)!=record['sha256'] or p.stat().st_size!=record['bytes']: raise ValueError('original artifact changed')
    return p


def slot_source(package,row):
    ref=package['goal007_manifest']['sources'][row['source_label']]
    plan=package['plans'][row['source_label']]
    record=[r for r in plan['identity']['schedule'] if r['schedule_index']==row['global_schedule_index']]
    full=package['full_manifest']['identity']['full_schedule'][row['global_schedule_index']]
    if len(record)!=1 or record[0]!=full: raise ValueError('schedule source differs')
    record=record[0]
    ledger=guard.read_json(ref['ledger_path']); entries=[e for e in ledger['entries'] if e['attempt_id']==row['attempt_id']]
    if len(entries)!=1 or object_hash(entries[0])!=row['source_entry_hash']: raise ValueError('selected attempt differs')
    entry=entries[0]
    for key in ('provider','input_id','scene_id','condition'):
        if record[key]!=row[key]: raise ValueError('slot identity differs: '+key)
    if entry['artifacts']!=row['artifacts'] or entry['status']!=row['generation_status']: raise ValueError('selected outcome differs')
    if row['input_sha256']!=record['input_sha256'] or hashlib.sha256(record['input_text'].encode()).hexdigest()!=row['input_sha256']:
        raise ValueError('input text hash differs')
    if Path(row['source_run'])!=Path(ref['enabled_directory']) or row['source_plan_identity']!=plan['identity_hash']:
        raise ValueError('canonical source run/plan differs')
    gt=plan['identity']['gt_by_scene'][row['scene_id']]
    if object_hash(gt)!=row['source_gt_descriptor_hash'] or gt['grid_sha256']!=row['gt_grid_sha256']:
        raise ValueError('GT descriptor differs')
    for name,record in row['artifacts'].items(): checked_artifact(ref['enabled_directory'],record)
    return gt,Limits(**plan['identity']['limits'])


def check_value(actual,expected,label):
    if actual is None or not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12):
        raise ValueError('independent metric disagreement: '+label)


def independent_checks(g,p,suite,shift,original=None):
    reports={}
    for frame in ('absolute','aligned'):
        world=p if frame=='absolute' else {tuple(c[i]+shift[i] for i in range(3)):m for c,m in p.items()}
        old,_=exact_oracle.independent_metrics(g,world)
        if original is not None: exact_oracle.verify_metrics(old,original[frame])
        check=independent.inspect(g,world); values=suite[frame]
        # Engine scalar IDs are fixed in METRIC_REGISTRY before data execution.
        for name,expected in [('occupancy_iou',old['occupancy_iou']),('occupancy_f1',old['occupancy_f1']),
            ('material_aware_iou',old['material_aware_iou'])]: check_value(values[name],expected,frame+'.'+name)
        ng,np_,c,m=check['g'],check['p'],check['C'],check['M']
        def ratio(a,b): return a/b if b else None
        expected={
            'gt_occupied_count':ng,'pred_occupied_count':np_,'occupancy_intersection':c,
            'occupancy_union':ng+np_-c,'occupancy_precision':ratio(c,np_),'occupancy_recall':ratio(c,ng),
            'excess_count':np_-c,'omission_count':ng-c,'volume_ratio':ratio(np_,ng),
            'relative_volume_error':ratio(np_-ng,ng),'exact_material_match_count':m,
            'conditional_exact_material_accuracy':ratio(m,c),'gt_material_recall':ratio(m,ng),
            'macro_gt_material_recall':check['macro_material_recall'],
            'gt_footprint_columns':check['gt_columns'],'pred_footprint_columns':check['pred_columns'],
            'common_footprint_columns':check['fp_intersection'],'union_footprint_columns':check['fp_union'],
            'footprint_iou':ratio(check['fp_intersection'],check['fp_union']) if check['fp_union'] else 1.0,
            'column_count_mae':ratio(check['column_absolute_error_sum'],check['fp_union']),
            'common_top_y_mae':ratio(check['common_top_y_absolute_error_sum'],check['fp_intersection']),
            'common_footprint_gt_coverage':ratio(check['fp_intersection'],check['gt_columns']),
            'common_footprint_pred_coverage':ratio(check['fp_intersection'],check['pred_columns']),
            'gt_surface_voxels':check['surface_g'],'pred_surface_voxels':check['surface_p'],
            'gt_components6':check['components_g']['count'],'pred_components6':check['components_p']['count'],
            'gt_largest_component_fraction6':check['components_g']['largest_fraction'],
            'pred_largest_component_fraction6':check['components_p']['largest_fraction']}
        for index,k in enumerate((1,2)):
            pr=ratio(check['surface_prediction_hits'][index],check['surface_p'])
            rc=ratio(check['surface_gt_hits'][index],check['surface_g'])
            f1=1.0 if not check['surface_g'] and not check['surface_p'] else 2*pr*rc/(pr+rc) if pr and rc else 0.0
            expected.update({f'surface_precision_at_{k}':pr,f'surface_recall_at_{k}':rc,f'surface_f1_at_{k}':f1})
        for key,info in check['pooled'].items():
            k,a=map(int,key.split('_')); anchor='0' if a==0 else 'half'
            expected[f'multiscale_iou_k{k}_anchor{anchor}']=info['iou']
        optional=set(metrics.SURFACE_METRIC_IDS)|set(metrics.CONNECTIVITY_METRIC_IDS)|{'gt_surface_voxels','pred_surface_voxels'}
        skipped=[]
        for name,value in expected.items():
            actual=values[name]
            if actual is None and value is not None and name in optional:
                reason=suite['details'][frame]['null_reasons'].get(name,'')
                if not reason.startswith(('METRIC_RESOURCE_LIMIT:','DEPENDENCY_UNAVAILABLE:')):
                    raise ValueError('unexplained optional metric NULL')
                skipped.append(name); continue
            if value is None:
                if actual is not None: raise ValueError('undefined metric must remainNULL: '+name)
            else: check_value(actual,value,frame+'.'+name)
        if c and not math.isclose(values['occupancy_f1'],2*values['occupancy_iou']/(1+values['occupancy_iou']),abs_tol=1e-12):
            raise ValueError('F1 IoU identity differs')
        reports[frame]={'strict_original_metrics_verified':original is not None,'independent':check,
                       'old_independent_metrics':old,'optional_checks_not_available':skipped,
                       'independent_scalar_checks':len(expected)+3-len(skipped)}
    if suite['aligned']['occupancy_iou']+1e-12<suite['absolute']['occupancy_iou']:
        raise ValueError('stored strict shift is worse than zero shift')
    dimensions=[max(c[i] for c in g)-min(c[i] for c in g)+1 for i in range(3)] if g else [0,0,0]
    l1=sum(abs(v) for v in shift); l2=math.sqrt(sum(v*v for v in shift))
    displacement={'shift_xyz':list(shift),'shift_l1':l1,'shift_l2':l2,'gt_occupied_dimensions_xyz':dimensions,
        'shift_abs_relative_xyz':[abs(shift[i])/dimensions[i] for i in range(3)] if g else None,
        'shift_l1_relative_extent_sum':l1/sum(dimensions) if g else None,
        'shift_l2_relative_bbox_diagonal':l2/math.sqrt(sum(v*v for v in dimensions)) if g else None,
        'occupancy_iou_gain':reports['aligned']['old_independent_metrics']['occupancy_iou']-reports['absolute']['old_independent_metrics']['occupancy_iou']}
    if suite['displacement']!=displacement: raise ValueError('independent displacement disagreement')
    reports['displacement']={'status':'PASS','checked_fields':list(displacement)}
    return reports


def evaluate(g,p,shift,original=None):
    with sample_budget():
        result=metrics.compute_metrics(g,p,shift)
        proof=independent_checks(g,p,result,shift,original)
    return result,proof


def adapt(row,result,proof,*,artificial=False):
    row['metrics']={f:result[f] for f in ('absolute','aligned')}
    row['null_reasons']={f:dict(result['details'][f]['null_reasons']) for f in ('absolute','aligned')}
    row['displacement']=result['displacement']; row['metric_provenance']=result['provenance']
    row['independent_checks']=proof; row['metric_status']='COMPUTED'
    details=result['details']
    if artificial:
        for frame in ('absolute','aligned'):
            for key in metrics.MATERIAL_METRIC_IDS:
                row['metrics'][frame][key]=None; row['null_reasons'][frame][key]='ARTIFICIAL_BASELINE_MATERIAL_NOT_INTERPRETED'
    return details


def flat_rows(rows):
    for row in rows:
        base={k:v for k,v in row.items() if k not in ('metrics','independent_checks','metric_provenance','null_reasons')}
        for frame,values in row['metrics'].items():
            base.update({frame+'__'+k:v for k,v in values.items()})
        base['null_reasons']=row['null_reasons']; yield base


def scene_order(package):
    scenes=sorted(package['full_manifest']['identity']['evaluation_scene_ids'])
    if (len(set(scenes))!=len(scenes) or set(scenes)!={r['scene_id'] for r in package['slots']}
            or set(scenes)!={r['scene_id'] for r in package['descriptions']}):
        raise ValueError('fixed evaluation scene identities differ from bound slots/descriptions')
    first='buildings_100_v4_building_034'
    if len(scenes)!=60 or first not in scenes: raise ValueError('fixed pilot scene absent')
    return [first]+[s for s in scenes if s!=first]


def capacity_diagnostic(oldrun,out):
    """Observed usage versus three fixed covariates; no capacity counterfactual."""
    source=oldrun/'phase3_exact_oracle/E_capacity_diagnostics.csv'
    with source.open() as f: data=list(csv.DictReader(f))
    if len(data)!=120: raise ValueError('fixed E120 capacity records required')
    panels=[('gt_occupied_voxels','GT occupied blocks'),('reference_operations','Reference cuboid operations'),
            ('spec_utf8_bytes','Exact specification UTF-8 bytes')]
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="640" viewBox="0 0 1140 640">',
        '<rect width="1140" height="640" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#20242a">',
        '<text x="30" y="24">E capacity diagnostic: observed output tokens (y), fixed plan cap 32768</text>',
        '<text x="30" y="44">Blue: valid; orange: truncation. Reference operations are not token estimates.</text>']
    for rownum,builder in enumerate(('openai','anthropic')):
        selected=[r for r in data if r['provider']==builder]
        if len(selected)!=60: raise ValueError('each E Builder60 records required')
        for col,(field,title) in enumerate(panels):
            left=55+col*375; top=85+rownum*270; width=300; height=180
            maxx=max(int(r[field]) for r in data)
            lines.extend([f'<text x="{left}" y="{top-12}">{builder}: {title}</text>',
                f'<path d="M {left} {top} V {top+height} H {left+width}" fill="none" stroke="#333"/>',
                f'<line x1="{left}" y1="{top}" x2="{left+width}" y2="{top}" stroke="#777" stroke-dasharray="4 4"/>',
                f'<text x="{left-45}" y="{top+4}">32768</text>',f'<text x="{left-12}" y="{top+height+4}">0</text>',
                f'<text x="{left}" y="{top+height+20}">0</text>',f'<text x="{left+width-45}" y="{top+height+20}">{maxx}</text>'])
            for r in selected:
                if not r['output_tokens_observed']: continue
                tokens=int(r['output_tokens_observed']); cap=int(r['output_token_cap'])
                if cap!=32768 or tokens<0 or tokens>cap: raise ValueError('unexpected observed output capacity')
                x=left+width*int(r[field])/maxx; y=top+height*(1-tokens/cap)
                color='#2774ae' if r['outcome_category']=='success' else '#ce672a'
                lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{color}" fill-opacity="0.65"/>')
    lines.extend(['<text x="30" y="620">CPU-only post-hoc diagnostic. Existing 120 outcomes; no new calls or causal fit.</text>','</g></svg>'])
    oldpipe.write_text(out/'E_capacity_diagnostic.svg','\n'.join(lines)+'\n')
    write_json(out/'E_capacity_plot_receipt.json',{'source_path':str(source),'source_sha256':sha(source),
        'outcomes':120,'x_covariates':[p[0] for p in panels],'output_cap':32768,'model_queries':0,
        'reference_bytes_not_token_estimates':True,'completion_counterfactuals_computed':False})


def evaluate_stage(package,oldrun,out,count,bound):
    scenes=scene_order(package)[:count]; selected=set(scenes)
    source_by_scene={r['scene_id']:r for r in package['descriptions']}
    oldbase=guard.read_json(oldrun/'phase1_baselines/baseline_per_building.json')
    rows=[]; base_rows=[]; material_rows=[]; qa=[]
    start=time.monotonic(); cpu=time.process_time()
    for scene_id in scenes:
        source=source_by_scene[scene_id]; scene=load_scene(source['gt_descriptor']['building_directory'])
        if scene.scene_id!=scene_id or object_hash(scene.grid.to_dict())!=source['gt_descriptor']['grid_sha256']:
            raise ValueError('GT world YXZ/XYZ/origin mismatch')
        g=scene.grid.as_world().voxels
        if not g: raise ValueError('actual GT must be nonempty')
        for saved in [r for r in package['slots'] if r['scene_id']==scene_id]:
            gt,limits=slot_source(package,saved)
            if gt['grid_sha256']!=source['gt_descriptor']['grid_sha256']: raise ValueError('GT join mismatch')
            keys=('global_schedule_index','scene_id','provider','description_model','condition','input_id','input_sha256',
                'attempt_id','original_attempt_id','selection_reason','source_label','source_plan_identity','source_entry_hash',
                'source_gt_descriptor_hash','gt_grid_sha256','status','failure_kind','usage_known','completed_request_outcome','http_status_code')
            row={k:saved.get(k) for k in keys}
            row.update(failure0_eligible=attempted_failure(saved),coordinate_system=COORDINATES,alignment_protocol=PROTOCOL,
                analysis_label='POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC',
                gt_voxels_sha256=gt['voxels_sha256'],gt_bbox_sha256=gt['bbox_sha256'],gt_storage_bbox=scene.bbox,
                gt_occupied_bbox=scene.grid.bbox(),artifact_sha256={k:v['sha256'] for k,v in saved['artifacts'].items()},
                original_metric_object_sha256=oldpipe.digest(saved['metrics']),implementation_hashes=bound['hashes'],
                specification_sha256=bound['specification_sha256'],empty_output=saved.get('empty_output'),metric_status='NOT_EVALUABLE',
                shift_xyz=None,displacement=None,metric_provenance=None,independent_checks=None)
            row['metrics'],row['null_reasons']=null_metrics('ORIGINAL_'+saved['status'].upper()+':'+str(saved.get('failure_kind')))
            if saved['status']=='success':
                shift=exact_oracle.validate_alignment(saved['metrics']); row['shift_xyz']=shift
                path=checked_artifact(saved['source_run'],saved['artifacts']['voxels.json'])
                pred=payload_to_grid(guard.read_json(path),limits)
                parsed=guard.read_json(checked_artifact(saved['source_run'],saved['artifacts']['parsed.json']))
                executed=execute(validate_actions(parsed['operations'],limits),limits)
                if executed.to_dict()!=pred.to_dict() or len(pred.voxels)!=saved['final_voxel_count']:
                    raise ValueError('saved prediction/actions/count mismatch')
                try: result,proof=evaluate(g,pred.as_world().voxels,shift,saved['metrics'])
                except (SampleResourceLimit,metrics.MetricResourceLimit,MemoryError) as exc:
                    reason='CPU_OR_MEMORY_RESOURCE_LIMIT:'+type(exc).__name__
                    row['metrics'],row['null_reasons']=null_metrics(reason); row['metric_status']='RESOURCE_LIMIT'
                else:
                    details=adapt(row,result,proof)
                    material_rows.append({'global_schedule_index':row['global_schedule_index'],'scene_id':scene_id,
                        'provider':row['provider'],'condition':row['condition'],'details':details})
            rows.append(row)
        for saved in [r for r in oldbase if r['scene_id']==scene_id]:
            keys=('prediction_id','scene_id','condition','description_model','input_id','input_sha256','source_sha256',
                'structure','gt_grid_sha256','gt_bbox_sha256','gt_voxels_sha256','size_rank_stratum','status')
            row={k:saved.get(k) for k in keys}
            row.update(coordinate_system=COORDINATES,alignment_protocol=PROTOCOL,artificial_material='stone',
                analysis_label='POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC',
                material_metrics_used=False,artifact_sha256={k:v['sha256'] for k,v in saved['artifacts'].items()},
                implementation_hashes=bound['hashes'],specification_sha256=bound['specification_sha256'],
                old_baseline_evaluation_sha256=sha(oldrun/'phase1_baselines/baseline_per_building.json'))
            shift=exact_oracle.validate_alignment({'metric_version':PROTOCOL,'alignment':saved['alignment']}); row['shift_xyz']=shift
            pred=baselines._load_prediction(saved,oldrun/'phase1_baselines/predictions',Limits())
            if saved['gt_grid_sha256']!=source['gt_descriptor']['grid_sha256']: raise ValueError('baseline GT differs')
            try: result,proof=evaluate(g,pred.voxels,shift)
            except (SampleResourceLimit,metrics.MetricResourceLimit,MemoryError) as exc:
                row['metrics'],row['null_reasons']=null_metrics('CPU_OR_MEMORY_RESOURCE_LIMIT:'+type(exc).__name__); row['metric_status']='RESOURCE_LIMIT'
            else:
                for frame in ('absolute','aligned'): check_value(result[frame]['occupancy_iou'],saved[frame+'_occupancy_iou'],'old baseline IoU')
                adapt(row,result,proof,artificial=True)
            base_rows.append(row)
        print(json.dumps({'scene_done':scene_id,'slots':len(rows),'baseline_predictions':len(base_rows),'cpu_seconds':time.process_time()-cpu}),flush=True)
        if time.process_time()-cpu>3600 or time.monotonic()-start>7200: raise SampleResourceLimit('STAGE_BUDGET_EXCEEDED_NO_FINAL_RECEIPT')
    rows.sort(key=lambda r:r['global_schedule_index']); base_rows.sort(key=lambda r:r['prediction_id'])
    prefix='new_metric_values_960' if count==60 else 'pilot_metric_values'
    compact_json(out/(prefix+'.json'),rows); write_csv(out/(prefix+'.csv'),list(flat_rows(rows)))
    compact_json(out/'baseline_new_metric_values.json',base_rows); write_csv(out/'baseline_new_metric_values.csv',list(flat_rows(base_rows)))
    compact_json(out/'material_and_geometry_details.json',material_rows)
    write_csv(out/'alignment_displacement.csv',[{**{k:r.get(k) for k in ('global_schedule_index','scene_id','provider','condition','description_model','status','metric_status','input_sha256','gt_grid_sha256')},
        **(r.get('displacement') or {}),'absolute_iou':r['metrics']['absolute']['occupancy_iou'],
        'aligned_iou':r['metrics']['aligned']['occupancy_iou'],
        'null_reason':None if r.get('displacement') else 'NO_VALID_MEASURABLE_RECONSTRUCTION'} for r in rows])
    confusion=[]; recalls=[]
    for r in material_rows:
        identity={k:r[k] for k in ('global_schedule_index','scene_id','provider','condition')}
        for frame in ('absolute','aligned'):
            for item in r['details'][frame]['confusion']: confusion.append({**identity,'frame':frame,**item})
            for item in r['details'][frame]['material_recalls']: recalls.append({**identity,'frame':frame,**item})
    write_csv(out/'exact_material_confusion.csv',confusion)
    write_csv(out/'gt_material_frequency_recall.csv',recalls)
    valid=[r for r in rows if r['status']=='success']
    computed=[r for r in valid if r['metric_status']=='COMPUTED']
    base_computed=[r for r in base_rows if r['metric_status']=='COMPUTED']
    optional_missing=sum(len(r['independent_checks'][frame]['optional_checks_not_available'])
        for r in computed+base_computed for frame in ('absolute','aligned'))
    report={'status':'PASS','buildings':count,'slots':len(rows),'original_status_counts':dict(Counter(r['status'] for r in rows)),
        'metric_status_counts':dict(Counter(r['metric_status'] for r in rows)),'baseline_status_counts':dict(Counter(r['metric_status'] for r in base_rows)),
        'baseline_predictions':len(base_rows),'original_valid_count':len(valid),'original_valid_metrics_checked':len(computed),
        'all_original_metrics_independently_checked_for_computed_success':True,
        'all_original_valid_metrics_checked':len(computed)==len(valid),
        'baseline_original_IoUs_checked':len(base_computed),'all_baseline_original_IoUs_identical':len(base_computed)==len(base_rows),
        'optional_metric_checks_unavailable':optional_missing,
        'all_surface_footprint_multiscale_independently_checked':len(computed)==len(valid) and len(base_computed)==len(base_rows) and optional_missing==0,
        'stored_strict_shift_only':True,'no_new_alignment_search':True,'world_YXZ_XYZ_control':True,'scene_ids':scenes,**ZERO}
    if len(rows)!=count*16 or len(base_rows)!=count*8: raise ValueError('scope counts changed')
    write_json(out/'metric_qa_report.json',report)
    return report


def run(stage,run_root,code_commit,spec_commit):
    if stage not in range(5): raise ValueError('stage0..4 required')
    root=safe_root(run_root,stage); bound=binding(code_commit,spec_commit)
    prev=previous(root,stage,bound); out=root/f'{stage}_{STAGES[stage]}'
    if out.exists() or (root/f'{stage}_STARTED.json').exists(): raise FileExistsError('preserve existing/partial stage')
    start=time.monotonic(); cpu=time.process_time(); created=False
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    try:
        with sample_budget(3600,7200), guard.locked_sources() as locks:
            package=guard.load_verified_sources()
            if oldpipe.digest(package)!=SOURCE_SHA: raise ValueError('source package changed')
            oldrun=verify_old()
            if stage==0: root.mkdir(exist_ok=False)
            write_json(root/f'{stage}_STARTED.json',{'stage':stage,'binding':bound,'at_utc':datetime.now(timezone.utc).isoformat()}); created=True
            out.mkdir(exist_ok=False)
            if stage==0:
                receipt=oldpipe.package_receipt(package); write_json(out/'SOURCE_BINDINGS.json',receipt)
                result={'status':'PASS','phase0_binding':'PASS','definitions_frozen':True,'input_kind':'verified_sources',
                    'specification_sha256':bound['specification_sha256'],'planned_scene_ids':sorted(scene_order(package)),
                    'prior_goal010_run':str(oldrun),'prior_goal010_manifest_sha256':OLD_MANIFEST_SHA,
                    'visibility_status':'NOT_VALIDATED_NO_VISIBILITY_CONCLUSION',**ZERO}
                write_json(out/'source_gate.json',result)
            elif stage in (1,2,3):
                result=evaluate_stage(package,oldrun,out,{1:1,2:10,3:60}[stage],bound)
            else:
                gate=guard.read_json(root/'0_source/source_gate.json')
                rows=guard.read_json(root/'3_full/new_metric_values_960.json'); bases=guard.read_json(root/'3_full/baseline_new_metric_values.json')
                result=statistics.summarize(rows,bases,gate=gate,specification_sha256=bound['specification_sha256'],registry=guard.read_json(REPORT/'METRIC_REGISTRY.json'))
                for name,key in [('metric_baselines_paired','paired_summary'),('metric_pairs','paired_rows'),('metric_exclusions','exclusions'),
                    ('metric_denominators','metric_denominators'),('reliability_outcomes','reliability'),('metric_rank_reversals','rank_reversals')]:
                    compact_json(out/(name+'.json'),result[key]); write_csv(out/(name+'.csv'),result[key])
                write_json(out/'statistics_metadata.json',result['metadata'])
                capacity_diagnostic(oldrun,out)
                result={'status':'PASS','counts':{k:len(v) for k,v in result.items() if isinstance(v,list)},**ZERO}
            after=guard.load_verified_sources()
            if oldpipe.digest(after)!=SOURCE_SHA: raise ValueError('source changed during stage')
            verify_old(); previous(root,stage,bound)
            if binding(code_commit,spec_commit)!=bound: raise ValueError('code changed during stage')
            receipt={'status':'PASS','stage':stage,'binding':bound,'previous':prev,'source_package_sha256':SOURCE_SHA,
                'source_before_after_unchanged':True,'source_lock_count':len(locks),'python_network_connections_disabled':True,
                'OS_level_isolation_guaranteed':False,'result':result,'products':oldpipe.products(root),
                'performance':{'elapsed_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,
                    'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'free_disk_bytes':shutil.disk_usage(root).free},**ZERO}
        write_json(root/f'{stage}_{STAGES[stage]}_RECEIPT.json',receipt)
        print(json.dumps({k:receipt[k] for k in ('status','stage','result','performance')}),flush=True)
        return receipt
    except Exception as exc:
        if created: write_json(root/f'{stage}_STOP.json',{'status':'STOP','error_type':type(exc).__name__,'reason':str(exc),**ZERO})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--stage',type=int,required=True)
    parser.add_argument('--run-root',required=True); parser.add_argument('--code-commit',required=True); parser.add_argument('--specification-commit',required=True)
    a=parser.parse_args(); run(a.stage,a.run_root,a.code_commit,a.specification_commit)


if __name__=='__main__': main()
