"""One final observation per fixed slot; all actual attempts remain auditable.

Only strict metrics missing from verified caches are computed. No source file,
generation state, approval, provider credential, or API is modified or queried.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from decimal import Decimal
import csv
import json
import math
import os
from pathlib import Path

from ...continuation_v1 import aggregate, runner as frozen_runner
from ...server_v2 import analysis_v4
from ...server_v2.alignment import AlignmentResourceLimit
from ...server_v2.core import evaluate_pair, grid_to_payload, payload_to_grid
from ...server_v2.data import load_scene, sha256_file
from ...server_v2.generation import GenerationError, object_hash, read_json, utc_now

VERSION = 'goal007_fixed60_one_operational_missing_v1'
ANALYSIS_LABEL = 'fixed60 analysis with one operational missing observation'
SOURCE_LABELS = ('parent160', 'original781', 'supplement3', 'supplement17')
EXPECTED_ATTEMPTS = dict(zip(SOURCE_LABELS, (160, 781, 3, 17)))
EXPECTED_SCHEDULES = {'parent160': list(range(160)), 'original781': list(range(160,960)),
                      'supplement3': list(range(940,960)), 'supplement17': list(range(943,960))}


def write_new_json(path, data):
    path = Path(path)
    payload = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def flatten(row, prefix=''):
    result = {}
    for key, value in row.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + '.'))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        else:
            result[name] = value
    return result


def write_new_csv(path, rows):
    rows = [flatten(row) for row in rows]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())


def validate_strict_metrics(metrics):
    if not isinstance(metrics, dict) or metrics.get('metric_version') != 'server_v2.2_strict_translation':
        raise GenerationError('goal007_strict_metric_version_required')
    alignment = metrics.get('alignment', {})
    if (alignment.get('mode') != 'strict_translation' or alignment.get('search_complete') is not True
            or alignment.get('globally_optimal_unrestricted') is not True
            or alignment.get('all_aligned_metrics_share_shift') is not True
            or alignment.get('rotation') is not False or alignment.get('scale') is not False
            or alignment.get('clipping') is not False or alignment.get('resource_limit_reached') is not False):
        raise GenerationError('goal007_exact_shared_translation_certificate_required')
    required = ('occupancy_iou','occupancy_f1','material_aware_iou','material_set_f1','dimension_mae',
                'total_normalized_repair_cost','add_count','delete_count','replace_count')
    for frame in ('absolute', 'aligned'):
        block = metrics.get(frame, {})
        for name in required:
            value = block.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise GenerationError('goal007_finite_nonnegative_metric_required')
            if name.endswith(('iou','f1')) and value > 1:
                raise GenerationError('goal007_bounded_overlap_required')


def load_sources(gate):
    if gate.get('status') != 'PASS':
        raise GenerationError('goal007_audit_gate_not_passed')
    sources = {}
    for label in SOURCE_LABELS:
        ref = gate['sources'][label]
        plan = read_json(Path(ref['plan_path'])); directory = Path(plan['identity']['enabled_directory'])
        evaluation_path = directory / 'evaluation.json'
        sources[label] = {'plan': plan, 'ledger': read_json(directory/'ledger.json'),
                          'plan_path': str(ref['plan_path']), 'directory': directory,
                          'evaluation': read_json(evaluation_path) if evaluation_path.exists() else None,
                          'evaluation_path': str(evaluation_path) if evaluation_path.exists() else None}
    manifest = read_json(Path(sources['original781']['plan']['identity']['manifest_path']))
    return manifest, sources


def _check_sources(manifest, sources):
    if set(sources) != set(SOURCE_LABELS):
        raise GenerationError('goal007_exact_four_sources_required')
    mi = manifest['identity']; full = mi['full_schedule']
    if (manifest.get('identity_hash') != object_hash(mi) or len(full) != 960
            or [r['schedule_index'] for r in full] != list(range(960))
            or len({(r['provider'],r['input_id']) for r in full}) != 960):
        raise GenerationError('goal007_exact_unique_fixed960_manifest_required')
    for label, source in sources.items():
        plan, ledger = source['plan'], source['ledger']; identity = plan['identity']
        if (plan['identity_hash'] != object_hash(identity) or ledger['plan_identity_hash'] != plan['identity_hash']
                or len(ledger['entries']) != EXPECTED_ATTEMPTS[label]):
            raise GenerationError('goal007_source_identity_or_attempt_count_mismatch')
        expected = [full[n] for n in EXPECTED_SCHEDULES[label]]
        aggregate._check_plan(plan, manifest, expected, mi['protocol_id'])
        for record, entry in zip(identity['schedule'], ledger['entries']):
            if (tuple(entry[k] for k in ('schedule_index','provider','input_id')) !=
                    tuple(record[k] for k in ('schedule_index','provider','input_id'))
                    or entry['request_payload_hash'] != object_hash(frozen_runner._payload(identity,record))):
                raise GenerationError('goal007_entry_payload_or_slot_mismatch')
    original = sources['original781']['ledger']['entries'][-1]
    retry = sources['supplement3']['ledger']['entries'][0]
    bplan = sources['supplement3']['plan']
    intent = bplan['identity']['intents'][0]
    if (original['schedule_index'] != 940 or original.get('http_status_code') != 401
            or original.get('usage') is not None or retry['status'] != 'success'
            or retry.get('usage') is None or retry.get('retry') is not True
            or retry.get('intent') != 'manual_auth_retry'
            or retry.get('source_attempt_id') != original['attempt_id']
            or intent.get('source_attempt_id') != original['attempt_id']
            or retry['request_payload_hash'] != original['request_payload_hash']):
        raise GenerationError('goal007_940_approved_retry_origin_required')
    missing = sources['supplement3']['ledger']['entries'][2]
    if (missing['schedule_index'] != 942 or missing.get('http_status_code') != 520
            or missing.get('usage') is not None or missing.get('artifacts') != {}
            or missing.get('failure_kind') != 'http_failure' or missing.get('status') != 'failed'):
        raise GenerationError('goal007_942_operational_unknown_required')
    return full


def integrate(manifest, sources):
    """Pure validated join, preserving cache scores without calculating statistics."""
    full = _check_sources(manifest,sources)
    attempts, candidate = [], {}
    for label in SOURCE_LABELS:
        source=sources[label]; plan=source['plan']; identity=plan['identity']; ledger=source['ledger']
        cached = aggregate._joined_rows(plan,source['evaluation'],ledger,manifest['identity']['protocol_id'])
        for local_index, entry in enumerate(ledger['entries']):
            record=identity['schedule'][local_index]; row=deepcopy(cached[local_index]); index=record['schedule_index']
            complete=frozen_runner._completed_response(identity,record,entry)
            usage_known=entry.get('usage') is not None
            if complete and not usage_known:
                raise GenerationError('goal007_completed_response_cannot_have_unknown_usage')
            classification=('operational_unknown' if not usage_known else 'success' if entry['status']=='success'
                            else 'model_output_failure' if complete and entry.get('failure_kind') in analysis_v4.DEFINITIVE_OUTPUT_FAILURES
                            else 'other_completed_failure' if complete else 'operational_error')
            history={'source_label':label,'source_plan_path':source['plan_path'],'source_plan_identity':plan['identity_hash'],
                     'source_run':str(source['directory']),'source_protocol':identity['protocol_id'],
                     'attempt_id':entry['attempt_id'],'global_schedule_index':index,
                     'provider':record['provider'],'builder_model':record['model'],'input_id':record['input_id'],
                     'scene_id':record['scene_id'],'description_model':record['description_model'],'condition':record['condition'],
                     'status':entry['status'],'failure_kind':entry.get('failure_kind'),'failure_class':classification,
                     'http_status':entry.get('http_status_code'),'http_status_code':entry.get('http_status_code'),'usage_known':usage_known,
                     'completed_request_outcome':complete,'usage':entry.get('usage'),
                     'charged_cost_usd':entry['charged_cost_usd'],'charged_input_tokens':entry['charged_input_tokens'],
                     'charged_output_tokens':entry['charged_output_tokens'],'billing':entry['billing'],
                     'manual_retry':entry.get('retry',False),'intent':entry.get('intent','first_attempt'),
                     'source_attempt_id':entry.get('source_attempt_id'),'request_payload_hash':entry['request_payload_hash'],
                     'gt_grid_sha256':identity['gt_by_scene'][record['scene_id']]['grid_sha256'],
                     'input_sha256':record['input_sha256'],'prompt_sha256':identity['prompt_sha256'],'schema_sha256':identity['schema_sha256'],
                     'artifacts':deepcopy(entry.get('artifacts',{})), 'selected_final_observation':index!=940 or label=='supplement3'}
            attempts.append(history)
            row.update({k:v for k,v in history.items() if k!='status'})
            row.update(schedule_index=index,attempted=True,original_attempt_id=entry['attempt_id'],
                       selected_attempt_id=entry['attempt_id'],attempt_history_ids=[entry['attempt_id']],
                       selection_reason='original_first_attempt',generation_status=entry['status'],
                       source_entry_hash=object_hash(entry),metric_cache_source=source.get('evaluation_path'),
                       evaluation_status='VERIFIED_CACHED' if row.get('metrics') is not None else
                           'ALIGNMENT_RESOURCE_LIMIT' if row['status']=='ALIGNMENT_RESOURCE_LIMIT' else
                           'PENDING_STRICT_EVALUATION' if entry['status']=='success' else 'NOT_APPLICABLE',
                       source_gt_descriptor_hash=object_hash(identity['gt_by_scene'][record['scene_id']]))
            if row.get('metrics') is not None:
                validate_strict_metrics(row['metrics'])
            candidate[(label,index)]=row
    if len(attempts)!=961 or len({(a['source_plan_identity'],a['attempt_id'],a['global_schedule_index'],a['provider'],a['input_id']) for a in attempts})!=961:
        raise GenerationError('goal007_exact_961_unique_attempt_history_required')
    counts=Counter(a['global_schedule_index'] for a in attempts)
    if counts!={**{n:1 for n in range(960)},940:2}:
        raise GenerationError('goal007_only_approved940_may_have_two_attempts')
    rows=[]
    for index in range(960):
        label='parent160' if index<160 else 'original781' if index<940 else 'supplement3' if index<943 else 'supplement17'
        row=candidate[(label,index)]
        if index==940:
            original=candidate[('original781',940)]
            row.update(original_attempt_id=original['attempt_id'],attempt_history_ids=[original['attempt_id'],row['attempt_id']],
                       selection_reason='approved_manual_auth_retry_after_401')
        if index==942:
            row.update(status='operational_unmeasurable',metrics=None,usage=None,response=None,completed_request_outcome=False,
                       evaluation_status='OPERATIONAL_UNMEASURABLE',response_model=None,response_id=None,
                       valid_nonempty_reconstruction=False)
        rows.append(row)
    validation=validate_slots(rows)
    return {'version':VERSION,'analysis_label':ANALYSIS_LABEL,'rows':rows,'attempts':attempts,'validation':validation}


def validate_slots(rows):
    if len(rows)!=960 or [r['global_schedule_index'] for r in rows]!=list(range(960)):
        raise GenerationError('goal007_exact960_global_order_required')
    if len({(r['provider'],r['input_id']) for r in rows})!=960:
        raise GenerationError('goal007_duplicate_provider_input')
    buildings={r['scene_id'] for r in rows}
    if len(buildings)!=60:
        raise GenerationError('goal007_exact60_buildings_required')
    for scene in buildings:
        subset=[r for r in rows if r['scene_id']==scene]
        expected={(p,c,d) for p in ('openai','anthropic') for c in ('O','J') for d in ('openai','anthropic','gemini')}
        expected|={(p,c,None) for p in ('openai','anthropic') for c in ('P','E')}
        if len(subset)!=16 or {(r['provider'],r['condition'],r['description_model']) for r in subset}!=expected:
            raise GenerationError('goal007_building_16_condition_cells_required')
    if [r['global_schedule_index'] for r in rows if r['status']=='operational_unmeasurable']!=[942]:
        raise GenerationError('goal007_exact_missing942_required')
    for row in rows:
        if row['status']!='success' and row.get('metrics') is not None:
            raise GenerationError('goal007_failure_metrics_must_be_null')
        if row.get('metrics') is not None: validate_strict_metrics(row['metrics'])
        if analysis_v4.attempted_failure(row) and (not row['usage_known'] or not row['completed_request_outcome']):
            raise GenerationError('goal007_failure0_requires_identified_known_usage_outcome')
    analysis_v4._validated(rows)
    return {'status':'PASS','fixed_slots':960,'unique_buildings':60,'builders':2,'description_providers':3,
            'condition_counts':dict(Counter(r['condition'] for r in rows)),'operational_missing_global_indices':[942],
            'all_slots_attempted':all(r['attempted'] for r in rows),'all_slots_measured':False,
            'analysis_label':ANALYSIS_LABEL,'retry940_counted_once':True}


def evaluate_missing(integrated, sources, destination, *, gt_loader=load_scene, evaluator=evaluate_pair):
    """Write an independent cache; no old evaluation/summary is ever updated."""
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=False)
    rows=integrated['rows']; grids={}; computed=[]; reused=[]; resource_limits=[]
    for row in rows:
        if row.get('metrics') is not None:
            reused.append(row['global_schedule_index']); continue
        if row['status']!='success': continue
        source=sources[row['source_label']]; identity=source['plan']['identity']
        descriptor=identity['gt_by_scene'][row['scene_id']]
        if row['scene_id'] not in grids:
            gt=gt_loader(descriptor['building_directory']).grid
            if object_hash(grid_to_payload(gt))!=descriptor['grid_sha256']:
                raise GenerationError('goal007_new_evaluation_gt_hash_mismatch')
            grids[row['scene_id']]=gt
        artifact=row['artifacts']['voxels.json']; path=Path(source['directory'])/artifact['path']
        if path.is_symlink() or not path.resolve().is_relative_to(Path(source['directory']).resolve()):
            raise GenerationError('goal007_voxel_artifact_path_escape')
        if sha256_file(path)!=artifact['sha256'] or path.stat().st_size!=artifact['bytes']:
            raise GenerationError('goal007_voxel_artifact_hash_mismatch')
        predicted=payload_to_grid(read_json(path))
        try:
            row['metrics']=evaluator(grids[row['scene_id']],predicted,alignment_mode='strict_translation')
            validate_strict_metrics(row['metrics']); row['evaluation_status']='VERIFIED_NEW_STRICT'
            row['valid_nonempty_reconstruction']=bool(predicted.voxels)
        except AlignmentResourceLimit as exc:
            row.update(status='ALIGNMENT_RESOURCE_LIMIT',metrics=None,evaluation_status='ALIGNMENT_RESOURCE_LIMIT',
                       evaluation_failure={'kind':'ALIGNMENT_RESOURCE_LIMIT','metadata':exc.metadata})
            resource_limits.append(row['global_schedule_index'])
        record={'version':VERSION,'plan_identity_hash':row['source_plan_identity'],'attempt_id':row['attempt_id'],
                'global_schedule_index':row['global_schedule_index'],'entry_hash':row['source_entry_hash'],
                'gt_descriptor_hash':row['source_gt_descriptor_hash'],'voxel_sha256':artifact['sha256'],
                'metric_version':'server_v2.2_strict_translation','metrics':row['metrics'],
                'evaluation_status':row['evaluation_status'],'evaluation_failure':row.get('evaluation_failure')}
        record['cache_hash']=object_hash(record)
        target=destination/f"{row['global_schedule_index']:04d}-{row['attempt_id']}.json"
        write_new_json(target,record); row['metric_cache_source']=str(target); computed.append(row['global_schedule_index'])
    integrated['validation']=validate_slots(rows)
    return {'computed_global_indices':computed,'reused_global_indices':reused,'computed_count':len(computed),
            'reused_count':len(reused),'resource_limit_global_indices':resource_limits,'old_cache_writes':0,
            'research_api_requests':0,'metric_version':'server_v2.2_strict_translation'}


def cost_audit(sources, attempts):
    prior=sources['parent160']['plan']['identity']['parent']['usage']
    historical_prior=Decimal(str(prior['cost_usd_upper_bound']))
    fixed=sum((Decimal(str(a['charged_cost_usd'])) for a in attempts),Decimal(0))
    unknown=[a for a in attempts if not a['usage_known']]
    combined=historical_prior+fixed
    calls=prior['calls']+len(attempts)
    c=sources['supplement17']; c_cost=sum((Decimal(str(e['charged_cost_usd'])) for e in c['ledger']['entries']),Decimal(0))
    chained=Decimal(str(c['plan']['identity']['parent']['usage']['cost_usd_upper_bound']))+c_cost
    if calls!=991 or combined!=chained or len(unknown)!=2 or {a['http_status'] for a in unknown}!={401,520}:
        raise GenerationError('goal007_historical_accounting_chain_mismatch')
    return {'status':'PASS','fixed_scope_attempts':len(attempts),'prior_smoke_calls':prior['calls'],'historical_calls':calls,
            'prior_budget_usd':str(historical_prior),'fixed_scope_budget_usd':str(fixed),'combined_budget_usd':str(combined),
            'supplement17_known_usage_usd':str(c_cost),'unknown_usage_holds':[
                {'attempt_id':a['attempt_id'],'global_schedule_index':a['global_schedule_index'],
                 'http_status':a['http_status'],'reservation_usd':str(a['charged_cost_usd'])} for a in unknown],
            'unknown_usage_holds_released':False,'actual_invoice_usd':None,
            'interpretation':'conservative known-usage calculation plus held worst-case unknown reservations, not an invoice',
            'new_research_requests':0}
