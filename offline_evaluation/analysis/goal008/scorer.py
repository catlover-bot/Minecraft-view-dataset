"""Missing-safe attribute fidelity; exploratory, source-bound, CPU-only."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

from offline_evaluation.server_v2.cohort_v4 import strict_saved_json
from offline_evaluation.server_v2.data import load_scene, sha256_file
from offline_evaluation.server_v2.generation import object_hash
from offline_evaluation.server_v2.core import grid_to_payload
from tools.evaluate_rebuild_metrics import coarse_material, normalize_block_type

SPECIFICATION_ID = 'goal008_missing_safe_attribute_fidelity_v1'
WEIGHTS = {'strict_material_f1':.30,'coarse_material_f1':.25,'dimension_accuracy':.25,'completeness':.20}
AIR = {'air','cave_air','void_air'}


def _invalid(value):
    text=json.dumps(value,sort_keys=True,ensure_ascii=False,default=str)
    return 'INVALID:'+hashlib.sha256(text.encode()).hexdigest()


def material_tokens(value):
    accepted=set(); invalid=set(); excluded_air=0
    if not isinstance(value,list):
        return accepted,invalid,{'list_status':'MISSING_OR_WRONG_TYPE','invalid_count':0,'air_items_excluded':0}
    for item in value:
        raw=item.get('name') if isinstance(item,dict) else item
        token=raw.strip().lower().removeprefix('minecraft:') if isinstance(raw,str) else ''
        if (not re.fullmatch('[a-z0-9_]+',token) or '_or_' in token
                or token in {'or','unknown','unspecified'}):
            invalid.add(_invalid(raw)); continue
        if token in AIR: excluded_air+=1
        else: accepted.add(token)
    return accepted,invalid,{'list_status':'LIST','invalid_count':len(invalid),'air_items_excluded':excluded_air}


def coarse_token(token):
    if token.startswith('INVALID:'): return token
    normalized=normalize_block_type(token); category=coarse_material(normalized)
    return category+':'+normalized if category=='OTHER' else category


def set_f1(predicted,target):
    if not target: raise ValueError('empty GT material target')
    return 2*len(predicted&target)/(len(predicted)+len(target))


def gt_material_target(grid):
    counts=Counter(grid.voxels.values()); total=sum(counts.values())
    if not total: raise ValueError('empty occupied GT')
    ordered=sorted(counts,key=lambda x:(-counts[x],x))
    selected=[x for x in ordered if counts[x]/total>=.03][:6] or ordered[:6]
    return set(selected),{'occupied_voxels':total,'top_k':6,'minimum_ratio':.03,
        'selected':[{'material':x,'count':counts[x],'fraction':counts[x]/total} for x in selected]}


def dimension_values(value,bbox):
    dims=value if isinstance(value,dict) else {}
    result={}; relative=[]
    for name,axis in [('width','x'),('depth','z'),('height','y')]:
        target=bbox[axis+'max']-bbox[axis+'min']+1
        if type(target) is not int or target<=0: raise ValueError('invalid GT bbox dimension')
        raw=dims.get(name); predicted=None; reason=None
        if type(raw) not in {int,float,Decimal}: reason='MISSING_OR_INVALID_TYPE'
        elif (isinstance(raw,Decimal) and not raw.is_finite()) or not 0<raw<=1_000_000 or not math.isfinite(raw):
            reason='NONFINITE_NONPOSITIVE_OR_OUT_OF_RANGE'
        else: predicted=max(1,round(raw))
        error=abs(predicted-target) if predicted is not None else None
        rel=error/target if error is not None else 1.0
        result[name]={'gt':target,'predicted':predicted,'missing_reason':reason,'absolute_error':error,'relative_error':rel}
        relative.append(rel)
    return max(0.,1.-min(1.,sum(relative)/3)),result


def score_record(description,grid,bbox):
    if not isinstance(description,dict): raise ValueError('O description must be a JSON object')
    target,gt_info=gt_material_target(grid)
    accepted,invalid,material_info=material_tokens(description.get('materials'))
    predicted=accepted|invalid
    strict=set_f1(predicted,target)
    coarse=set_f1({coarse_token(x) for x in predicted},{coarse_token(x) for x in target})
    dim_score,dims=dimension_values(description.get('dimensions_estimate'),bbox)
    def has_two_strings(value):
        return isinstance(value,list) and sum(isinstance(x,str) and bool(x.strip()) for x in value)>=2
    summary=description.get('summary')
    fields={'summary':isinstance(summary,str) and len(summary.strip())>=30,'materials':len(accepted)>=2,
        'dimensions':all(v['predicted'] is not None for v in dims.values()),
        'elements':has_two_strings(description.get('elements')),'rebuild_hints':has_two_strings(description.get('rebuild_hints'))}
    scores={'strict_material_f1':strict,'coarse_material_f1':coarse,'dimension_accuracy':dim_score,
            'completeness':sum(fields.values())/5}
    scores['composite']=sum(scores[k]*v for k,v in WEIGHTS.items())
    occupied={a:max(c[i] for c in grid.voxels)-min(c[i] for c in grid.voxels)+1 for i,a in enumerate('xyz')}
    return {'scores':scores,'gt_material_target':gt_info,'pred_materials':sorted(accepted),
        'invalid_material_tokens':sorted(invalid),'material_input_status':material_info,'dimensions':dims,
        'occupied_gt_dimensions_xyz':occupied,'completeness_fields':fields,
        'missing_reasons':{k:v['missing_reason'] for k,v in dims.items() if v['missing_reason']},
        'interpretation':'new exploratory CPU-derived attribute fidelity proxy; not conventional description quality'}


def compute_scores(source_rows,*,definition_sha256,source_loader=load_scene,fixed_design=True):
    if fixed_design:
        keys={(r['scene_id'],r['description_provider']) for r in source_rows}
        scenes={r['scene_id'] for r in source_rows}
        if len(source_rows)!=180 or len(scenes)!=60 or keys!={(s,p) for s in scenes for p in ('openai','anthropic','gemini')}:
            raise ValueError('exact180 source descriptions required')
    grids={}; records=[]
    # Check every source before calculating any score; no partial-score publication.
    for row in source_rows:
        if sha256_file(row['source_path'])!=row['source_sha256']: raise ValueError('description source hash mismatch')
        gt=row['gt_descriptor']
        for suffix,key in [('gt/bbox.json','bbox_sha256'),('gt/voxels.npy','voxels_sha256')]:
            if sha256_file(Path(gt['building_directory'])/suffix)!=gt[key]: raise ValueError('GT file hash mismatch')
    for row in source_rows:
        scene=row['scene_id']; gt=row['gt_descriptor']
        if scene not in grids:
            loaded=source_loader(gt['building_directory'])
            if object_hash(grid_to_payload(loaded.grid))!=gt['grid_sha256']: raise ValueError('GT grid hash mismatch')
            grids[scene]=loaded
        description=strict_saved_json(Path(row['source_path']).read_text(encoding='utf-8'))
        scored=score_record(description,grids[scene].grid,grids[scene].bbox)
        records.append({**row,'gt_bbox_sha256':gt['bbox_sha256'],'gt_voxels_sha256':gt['voxels_sha256'],
            'scoring_specification_id':SPECIFICATION_ID,'definition_sha256':definition_sha256,
            'scorer_sha256':sha256_file(__file__),**scored})
    return records
