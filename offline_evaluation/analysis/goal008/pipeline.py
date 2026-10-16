"""Freeze and run Goal008 Phase0/1 before any actual example selection."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time

from . import audit, scorer, statistics
from ..goal007.integration import write_new_json
from ...server_v2.data import sha256_file
from ...server_v2.generation import object_hash, utc_now

ROOT=audit.ROOT
REPORT=ROOT/'reports/server_only/goal008'
FROZEN_FILES=[f'offline_evaluation/analysis/goal008/{n}.py' for n in ['__init__','audit','scorer','statistics','pipeline']]+[
    'tools/evaluate_description_quality.py','tools/evaluate_rebuild_metrics.py',
    *[f'reports/server_only/goal008/{n}' for n in ['definition.md','stats_definition.md','example_definition.md']]]


def verify_committed(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('explicit code commit required')
    hashes={}
    for name in FROZEN_FILES:
        h=sha256_file(ROOT/name)
        stored=subprocess.run(['git','show',f'{commit}:{name}'],cwd=ROOT,capture_output=True)
        if stored.returncode or hashlib.sha256(stored.stdout).hexdigest()!=h:
            raise ValueError('code/specification commit mismatch: '+name)
        hashes[name]=h
    return hashes


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fields,lineterminator='\n'); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def augment_slots(slots,sources):
    """Append audited metadata in memory; never modify the original slots."""
    original=object_hash(slots); result=deepcopy(slots)
    index={(r['scene_id'],r['description_provider']):r for r in sources}
    for row in result:
        if row['condition']!='O': continue
        source=index[(row['scene_id'],row['description_model'])]
        observed=source['builder_observations'][row['provider']]
        if (row['input_id']!=source['input_id'] or row['input_sha256']!=source['input_sha256']
            or row['gt_grid_sha256']!=source['gt_descriptor']['grid_sha256']
            or row['source_gt_descriptor_hash']!=source['gt_descriptor_sha256']
            or row['attempt_id']!=observed['attempt_id'] or row['global_schedule_index']!=observed['global_schedule_index']):
            raise ValueError('audited O-source augmentation mismatch')
        added={'source_sha256':source['source_sha256'],'gt_bbox_sha256':source['gt_descriptor']['bbox_sha256'],
               'gt_voxels_sha256':source['gt_descriptor']['voxels_sha256']}
        if any(k in row and row[k]!=v for k,v in added.items()): raise ValueError('conflicting existing source hash')
        row.update(added)
    if object_hash(slots)!=original: raise AssertionError('original slots mutated')
    return result


def descriptive_scores(scores):
    import numpy as np
    result=[]
    for provider in [None,'openai','anthropic','gemini']:
        selected=[r for r in scores if provider is None or r['description_provider']==provider]
        for feature in statistics.FEATURES:
            values=[r['scores'][feature] for r in selected if r['scores'][feature] is not None]
            result.append({'description_provider':provider,'feature':feature,'descriptions':len(selected),
                'buildings':len({r['scene_id'] for r in selected}),'valid_values':len(values),
                'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None,
                'minimum':min(values) if values else None,'maximum':max(values) if values else None,
                'zero_values':sum(v==0 for v in values),'one_values':sum(v==1 for v in values),'unique_values':len(set(values)),
                'interpretation':'new exploratory CPU-derived attribute fidelity proxy; not conventional description quality'})
    return result


def run_phase1(audit_gate,destination,code_commit):
    started=time.monotonic(); out=Path(destination).absolute()
    if out.exists() or not out.is_relative_to(Path('/root/minecraft-runs')): raise ValueError('new research output directory required')
    if shutil.disk_usage(out.parent).free<10*1024**3: raise ValueError('10GiB free disk required')
    hashes=verify_committed(code_commit)
    gate=audit.verify_report(audit_gate)
    old_manifest=audit.read_json(audit.GOAL007/'MANIFEST.json')
    plans={k:audit.read_json(v['plan_path']) for k,v in old_manifest['sources'].items()}
    with audit.old_audit.source_locks(plans),audit.old_audit.no_python_network():
        audit.old_audit.process_audit()
        audit.verify_report(gate)
        out.mkdir(parents=True,exist_ok=False)
        definition_hash=sha256_file(REPORT/'definition.md'); stats_hash=statistics.definition_sha256()
        sources=audit.read_json(gate['description_sources_path'])
        slots=audit.read_json(audit.GOAL007/'data/final_slots_960.json')
        frozen={'status':'FROZEN_BEFORE_SCORING','at_utc':utc_now(),'code_commit':code_commit,
                'code_and_definition_hashes':hashes,'source_gate_sha256':sha256_file(audit_gate),
                'scorer_definition_sha256':definition_hash,'statistics_definition_sha256':stats_hash,
                'example_definition_sha256':sha256_file(REPORT/'example_definition.md'),'research_api_requests_sent':0}
        write_new_json(out/'definitions_frozen.json',frozen)
        print('GOAL008 Phase0 PASS; compute180 source-bound missing-safe proxies',flush=True)
        scores=scorer.compute_scores(sources,definition_sha256=definition_hash)
        augmented=augment_slots(slots,sources)
        statistics_gate={'status':'PASS','input_kind':'verified_sources','phase0_binding':'PASS','definitions_frozen':True,
                         'matched_description_count':len(scores),'definition_sha256':stats_hash}
        print('GOAL008 run360 fixed exploratory associations;2000 building bootstrap draws',flush=True)
        associations=statistics.analyze_correlations(scores,augmented,audit_gate=statistics_gate,definition_sha256=stats_hash)
        groups=descriptive_scores(scores)
        write_new_json(out/'proxy_scores_180.json',scores)
        write_csv(out/'proxy_scores_180.csv',[{**{k:r[k] for k in ['scene_id','description_provider','description_model_id','source_path','source_sha256','input_id','input_sha256','gt_bbox_sha256','gt_voxels_sha256','scoring_specification_id','scorer_sha256','definition_sha256']},**r['scores'],'missing_reasons':r['missing_reasons'],'invalid_material_count':len(r['invalid_material_tokens'])} for r in scores])
        write_new_json(out/'exploratory_associations.json',associations)
        write_csv(out/'exploratory_associations_360.csv',associations['associations'])
        write_csv(out/'proxy_descriptive.csv',groups)
        write_new_json(out/'joined_descriptions_180.json',associations['joined_descriptions'])
        write_csv(out/'source_map_180.csv',[{k:r[k] for k in ['scene_id','description_provider','description_model_id','source_path','source_sha256','input_id','input_sha256','j_input_id','j_input_sha256','gt_bbox_path','gt_voxels_path','gt_descriptor_sha256']} for r in sources])
        audit.verify_report(gate); audit.old_audit.process_audit()
        if verify_committed(code_commit)!=hashes: raise ValueError('analysis code changed during execution')
        products={p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
        final={'status':'PHASE0_PHASE1_PASS','version':'goal008_phase1_v1','created_at_utc':utc_now(),'created_code_commit':code_commit,
            'code_and_definition_hashes':hashes,'source_gate_path':str(Path(audit_gate).absolute()),'source_gate_sha256':sha256_file(audit_gate),
            'goal007_manifest_sha256':audit.MANIFEST_SHA256,'goal007_slots_sha256':sha256_file(audit.GOAL007/'data/final_slots_960.json'),
            'goal007_comparisons_sha256':sha256_file(audit.GOAL007/'results/comparisons_20.csv'),
            'matched_descriptions':len(scores),'source_bound_builder_observations':360,'independent_buildings':60,
            'planned_associations':len(associations['associations']),'association_status_counts':associations['status_counts'],
            'historical_RQ2_status':'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES','new_analysis':'exploratory CPU-derived attribute fidelity proxy',
            'source_inventories_before_after_unchanged':True,'research_api_requests_sent':0,'api_key_queries':0,'model_queries':0,
            'human_experiments':0,'checkpoint_mutations':0,'products':products,
            'environment':{k:importlib.metadata.version(k) for k in ['numpy','scipy']},
            'performance':{'seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'free_disk_bytes':shutil.disk_usage(out).free}}
        write_new_json(out/'PHASE1_MANIFEST.json',final)
        print(json.dumps({k:final[k] for k in ['status','matched_descriptions','planned_associations','association_status_counts','performance','research_api_requests_sent']}),flush=True)
        return final


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-gate',required=True);p.add_argument('--destination',required=True);p.add_argument('--code-commit',required=True)
    a=p.parse_args(argv)
    try: run_phase1(a.audit_gate,a.destination,a.code_commit)
    except Exception as exc:
        print(json.dumps({'status':'STOP','error_type':type(exc).__name__,'safe_reason':str(exc) if isinstance(exc,(ValueError,audit.AuditError)) else None,'research_api_requests_sent':0}),flush=True)
        return 5
    return 0


if __name__=='__main__': raise SystemExit(main())
