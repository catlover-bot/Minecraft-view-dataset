"""Incremental CPU metrics and the fixed960 building-level analysis entrance.

This module contains no generation or credential code. Parent160 measurements
are read at their frozen hashes; only newly completed continuation outputs need
metrics. A complete plan is not itself evidence of complete measurements.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import csv
from statistics import mean

from ..server_v2 import analysis_v4 as analysis
from ..server_v2.core import evaluate_pair, grid_to_payload, payload_to_grid
from ..server_v2.data import load_scene, sha256_file
from ..server_v2.alignment import AlignmentResourceLimit
from ..server_v2.generation import GenerationError, object_hash, read_json, write_json, utc_now, digest

STAGES = ('api_response_received', 'natural_stop', 'json_success', 'schema_success', 'executor_success',
          'operation_count', 'final_voxel_count', 'empty_output')


def _row(identity, record, entry):
    from . import runner
    row = {k: record.get(k) for k in ('provider', 'scene_id', 'condition', 'description_model', 'input_id')}
    row.update(provenance='new_implementation', protocol_id=identity['protocol_id'],
               source_run_protocol_id=identity['protocol_id'], semantic_protocol_id=identity['semantic_protocol_id'],
               schedule_index=record['schedule_index'], input_sha256=record['input_sha256'],
               prompt_sha256=identity['prompt_sha256'], schema_sha256=identity['schema_sha256'],
               gt_grid_sha256=identity['gt_by_scene'][record['scene_id']]['grid_sha256'],
               attempt_id=entry['attempt_id'] if entry else None, attempted=entry is not None,
               status=entry['status'] if entry else 'NOT_RUN', failure_kind=entry.get('failure_kind') if entry else None,
               completed_request_outcome=runner._completed_response(identity, record, entry),
               metrics=None, valid_nonempty_reconstruction=False,
               response_model=entry.get('response_model') if entry else None,
               started_at_utc=entry.get('started_at_utc') if entry else None,
               finished_at_utc=entry.get('finished_at_utc') if entry else None)
    row.update({name: entry.get(name) if entry else None for name in STAGES})
    return row


def evaluate_verified(plan, directory, ledger, gt_by_scene=None):
    """Called under run locks after the existing artifact validator, exactly once.

    Existing cache rows are bound to immutable entries, GT, and plan/code identity.
    Cached quality failures also stay explicit; a corrupt cache fails closed.
    """
    from . import runner
    identity, directory = plan['identity'], Path(directory)
    if not directory.is_dir():
        raise GenerationError('run_not_enabled; CPU_evaluation_cannot_create_enabled')
    cache_path = directory / 'evaluation.json'
    cache = read_json(cache_path) if cache_path.exists() else None
    old = {}
    if cache is not None:
        core = {k: v for k, v in cache.items() if k != 'cache_hash'}
        if cache.get('plan_identity_hash') != plan['identity_hash'] or cache.get('cache_hash') != object_hash(core):
            raise GenerationError('incremental_evaluation_cache_identity_mismatch')
        old = {row['schedule_index']: row for row in cache['rows']}
        if len(old) != len(cache['rows']):
            raise GenerationError('incremental_evaluation_cache_duplicate')
    grids = {} if gt_by_scene is None else dict(gt_by_scene)
    rows, calculated, reused = [], 0, 0
    for index, record in enumerate(identity['schedule']):
        entry = ledger['entries'][index] if index < len(ledger['entries']) else None
        row = _row(identity, record, entry)
        source = {'plan_identity_hash': plan['identity_hash'], 'entry_hash': object_hash(entry),
                  'gt_hash': object_hash(identity['gt_by_scene'][record['scene_id']])}
        previous = old.get(record['schedule_index'])
        if previous and previous.get('attempted'):
            if previous.get('cache_source') != source:
                raise GenerationError('previous_evaluated_attempt_changed')
            # Verify current canonical row metadata independently from cached metrics.
            for name, value in row.items():
                if name not in {'metrics', 'status', 'valid_nonempty_reconstruction'} and previous.get(name) != value:
                    raise GenerationError('cached_evaluation_outcome_identity_mismatch')
            if previous['status'] != row['status'] and not (previous['status'] == 'ALIGNMENT_RESOURCE_LIMIT' and row['status'] == 'success'):
                raise GenerationError('cached_evaluation_status_mismatch')
            row = deepcopy(previous)
            reused += int(entry is not None and entry['status'] == 'success')
        elif entry and entry['status'] == 'success':
            descriptor = identity['gt_by_scene'][record['scene_id']]
            if record['scene_id'] not in grids:
                grids[record['scene_id']] = load_scene(descriptor['building_directory']).grid
            gt = grids[record['scene_id']]
            if object_hash(grid_to_payload(gt)) != descriptor['grid_sha256']:
                raise GenerationError('evaluation_gt_hash_mismatch')
            pred = payload_to_grid(read_json(runner._artifact(directory, entry['artifacts']['voxels.json'])))
            row['valid_nonempty_reconstruction'] = bool(pred.voxels)
            try:
                row['metrics'] = evaluate_pair(gt, pred, alignment_mode='strict_translation')
            except AlignmentResourceLimit as exc:
                row.update(status='ALIGNMENT_RESOURCE_LIMIT', generation_status=entry['status'],
                           evaluation_failure={'kind': 'ALIGNMENT_RESOURCE_LIMIT', 'metadata': exc.metadata})
            calculated += 1
        row['cache_source'] = source
        rows.append(row)
    envelope = {'version': 'continuation_incremental_evaluation_v1', 'plan_identity_hash': plan['identity_hash'], 'rows': rows}
    envelope['cache_hash'] = object_hash(envelope)
    summary = analysis.summarize_v4(rows, smoke=True)
    summary.update(runner.execution_summary(plan, ledger))
    summary.update(analysis_scope='remaining50 continuation descriptive; use aggregate for fixed60 final inference',
                   metrics_computed_this_call=calculated, metrics_reused_this_call=reused,
                   quality_evaluated_inputs=sum(r['metrics'] is not None for r in rows),
                   stage_counts={name: sum(r.get(name) is True for r in rows) for name in STAGES if name not in {'operation_count', 'final_voxel_count'}},
                   research_api_requests_sent_by_evaluation=0)
    write_json(cache_path, envelope); write_json(directory / 'summary.json', summary)
    return summary


def _check_plan(plan, manifest, expected_schedule, semantic):
    identity = plan['identity']
    if plan['identity_hash'] != object_hash(identity) or identity['schedule'] != expected_schedule:
        raise GenerationError('aggregate_plan_or_schedule_identity_mismatch')
    if identity['manifest_identity_hash'] != manifest['identity_hash']:
        raise GenerationError('aggregate_manifest_binding_mismatch')
    for name in ('prompt_sha256', 'schema_sha256'):
        if identity[name] != manifest['identity'][name]:
            raise GenerationError('aggregate_prompt_or_schema_mismatch')
    if 'semantic_protocol_id' in identity and identity['semantic_protocol_id'] != semantic:
        raise GenerationError('aggregate_semantic_protocol_mismatch')
    for record in expected_schedule:
        if (record['protocol_id'] != semantic or record['prompt_sha256'] != identity['prompt_sha256']
                or record['schema_sha256'] != identity['schema_sha256']
                or digest(record['input_text'].encode()) != record['input_sha256']):
            raise GenerationError('aggregate_input_semantics_mismatch')
        if identity['gt_by_scene'][record['scene_id']] != manifest['identity']['gt_by_scene'][record['scene_id']]:
            raise GenerationError('aggregate_GT_identity_mismatch')


def _joined_rows(plan, evaluation, ledger, semantic):
    from . import runner
    identity = plan['identity']
    if evaluation is None:
        evaluation = {'plan_identity_hash': plan['identity_hash'], 'rows': []}
    if evaluation.get('plan_identity_hash') != plan['identity_hash']:
        raise GenerationError('aggregate_evaluation_identity_mismatch')
    if 'cache_hash' in evaluation and evaluation['cache_hash'] != object_hash({k: v for k, v in evaluation.items() if k != 'cache_hash'}):
        raise GenerationError('aggregate_evaluation_cache_hash_mismatch')
    saved = {}
    for row in evaluation['rows']:
        key = (row['provider'], row['input_id'])
        if key in saved:
            raise GenerationError('aggregate_duplicate_provider_input')
        saved[key] = row
    schedule = identity['schedule']
    expected_keys = {(r['provider'], r['input_id']) for r in schedule}
    if not saved.keys() <= expected_keys:
        raise GenerationError('aggregate_unplanned_evaluation_row')
    if len(ledger['entries']) > len(schedule):
        raise GenerationError('aggregate_extra_attempts')
    output = []
    for index, record in enumerate(schedule):
        entry = ledger['entries'][index] if index < len(ledger['entries']) else None
        key = (record['provider'], record['input_id'])
        row = deepcopy(saved.get(key))
        if entry and (entry['provider'], entry['input_id'], entry['schedule_index']) != (*key, record['schedule_index']):
            raise GenerationError('aggregate_ledger_schedule_mismatch')
        if row is None:
            # A planned missing score is explicit, never a fabricated measurement.
            derived_identity = {**identity, 'semantic_protocol_id': semantic}
            row = _row(derived_identity, record, entry)
            row['evaluation_status'] = 'MISSING' if entry and entry['status'] == 'success' else 'NOT_REQUIRED'
        for name in ('provider', 'scene_id', 'condition', 'description_model', 'input_id'):
            if row.get(name) != record.get(name):
                raise GenerationError('aggregate_row_input_identity_mismatch')
        if (row.get('protocol_id') != identity['protocol_id'] or row.get('attempted') != (entry is not None)
                or row.get('attempt_id') != (entry['attempt_id'] if entry else None)):
            raise GenerationError('aggregate_row_run_or_attempt_mismatch')
        for stage in STAGES:
            if row.get(stage) != (entry.get(stage) if entry else None):
                raise GenerationError('aggregate_stage_flag_mismatch')
        if row.get('completed_request_outcome') != runner._completed_response(identity, record, entry):
            raise GenerationError('aggregate_completed_response_mismatch')
        expected_status = entry['status'] if entry else 'NOT_RUN'
        if row['status'] != expected_status and not (row['status'] == 'ALIGNMENT_RESOURCE_LIMIT' and expected_status == 'success'):
            raise GenerationError('aggregate_row_status_mismatch')
        if row.get('failure_kind') != (entry.get('failure_kind') if entry else None):
            raise GenerationError('aggregate_row_failure_mismatch')
        for field, value in (('semantic_protocol_id', semantic), ('input_sha256', record['input_sha256']),
                             ('prompt_sha256', identity['prompt_sha256']), ('schema_sha256', identity['schema_sha256']),
                             ('gt_grid_sha256', identity['gt_by_scene'][record['scene_id']]['grid_sha256'])):
            if field in row and row[field] != value:
                raise GenerationError('aggregate_row_semantic_or_GT_mismatch')
            row[field] = value
        row.update(source_run_protocol_id=identity['protocol_id'], schedule_index=record['schedule_index'],
                   response_model=entry.get('response_model') if entry else None,
                   started_at_utc=entry.get('started_at_utc') if entry else None,
                   finished_at_utc=entry.get('finished_at_utc') if entry else None,
                   usage_known=bool(entry and entry.get('usage') is not None))
        output.append(row)
    return output


def aggregate_verified(plan, parent_plan, parent_eval, continuation_eval, parent_ledger, ledger, manifest=None):
    """Pure join of already artifact-validated inputs; does no metric replay."""
    manifest = read_json(plan['identity']['manifest_path']) if manifest is None else manifest
    mi = manifest['identity']; full = mi['full_schedule']; semantic = mi['protocol_id']
    if manifest['identity_hash'] != object_hash(mi) or len(full) != 960 or len(mi['evaluation_scene_ids']) != 60:
        raise GenerationError('aggregate_requires_original_fixed960_manifest')
    if [r['schedule_index'] for r in full] != list(range(960)) or len({(r['provider'], r['input_id']) for r in full}) != 960:
        raise GenerationError('aggregate_global_index_or_semantic_duplicate')
    _check_plan(parent_plan, manifest, full[:160], semantic); _check_plan(plan, manifest, full[160:], semantic)
    if plan['identity']['parent']['plan_identity_hash'] != parent_plan['identity_hash']:
        raise GenerationError('aggregate_wrong_parent')
    rows = _joined_rows(parent_plan, parent_eval, parent_ledger, semantic) + _joined_rows(plan, continuation_eval, ledger, semantic)
    rows = analysis._validated(rows)
    complete = all(r['usage_known'] and r['completed_request_outcome'] and
                   ((r['status'] == 'success' and r['metrics'] is not None) or analysis.attempted_failure(r)) for r in rows)
    summary = analysis.summarize_v4(rows, smoke=not complete)
    comparisons = []
    for family in analysis.comparison_families():
        for comparison in family['comparisons']:
            pairs = analysis._paired(rows, comparison)
            all_pairs = analysis._paired(rows, comparison, True) if comparison['metric'] in analysis.BOUNDED_IOU else []
            comparisons.append({**comparison, 'family': family['id'], 'common_valid_buildings': sorted({r['scene_id'] for r in pairs}),
                                'common_valid_pair_count': len(pairs), 'all_attempted_failure0_buildings': sorted({r['scene_id'] for r in all_pairs})})
    by_description = []
    for provider in analysis.PROVIDERS:
        for condition in ('O', 'J'):
            for description in ('openai', 'anthropic', 'gemini'):
                chosen = [r for r in rows if (r['provider'], r['condition'], r['description_model']) == (provider, condition, description)]
                by_description.append({'provider': provider, 'condition': condition, 'description_model': description,
                                       'planned': len(chosen), 'valid_only': analysis._descriptive_metric(chosen, 'aligned_occupancy_iou'),
                                       'failure0_eligible': sum(analysis.attempted_failure(r) for r in chosen)})
    bounded = []
    for provider in analysis.PROVIDERS:
        for condition in analysis.CONDITIONS:
            chosen = [r for r in rows if r['provider'] == provider and r['condition'] == condition]
            for metric in sorted(analysis.BOUNDED_IOU):
                grouped = {}
                for row in chosen:
                    value = analysis.bounded_all_attempted_value(row, metric)
                    if value is not None:
                        grouped.setdefault(row['scene_id'], []).append(value)
                bounded.append({'provider': provider, 'condition': condition, 'metric': metric,
                                'valid_only': analysis._descriptive_metric(chosen, metric),
                                'all_attempted_failure0_rows': sum(len(v) for v in grouped.values()),
                                'all_attempted_failure0_buildings': sorted(grouped),
                                'all_attempted_failure0_equal_building_mean': mean(mean(v) for v in grouped.values()) if grouped else None,
                                'excluded_not_run_or_unmeasurable': len(chosen) - sum(len(v) for v in grouped.values())})
    e_sets = {p: {r['scene_id'] for r in rows if r['provider'] == p and r['condition'] == 'E' and r['status'] == 'success' and r['metrics'] is not None} for p in analysis.PROVIDERS}
    e = {p: {'valid_scenes': sorted(e_sets[p]), 'truncation_scenes': sorted(r['scene_id'] for r in rows if r['provider'] == p and r['condition'] == 'E' and r.get('failure_kind') == 'truncation'),
             'absolute_shape_exact_scenes': sorted(r['scene_id'] for r in rows if r['provider'] == p and r['condition'] == 'E' and analysis._metric(r, 'absolute_occupancy_iou') == 1),
             'absolute_labelled_exact_scenes': sorted(r['scene_id'] for r in rows if r['provider'] == p and r['condition'] == 'E' and analysis._metric(r, 'absolute_material_aware_iou') == 1)} for p in analysis.PROVIDERS}
    for provider in analysis.PROVIDERS:
        e[provider]['valid_nonexact_labelled_scenes'] = sorted(e_sets[provider] - set(e[provider]['absolute_labelled_exact_scenes']))
    summary.update(analysis_scope='FINAL_FIXED60' if complete else 'INTERIM_DESCRIPTIVE_FIXED60',
                   fixed60_complete=complete, planned_inputs=960, planned_buildings=60,
                   attempted_inputs=sum(r['attempted'] for r in rows), not_run_inputs=sum(not r['attempted'] for r in rows),
                   common_valid_denominators=comparisons, bounded_IoU_denominators=bounded, OJ_description_provider_tables=by_description,
                   E_diagnostics=e, E_common_valid_scenes=sorted(set.intersection(*e_sets.values())),
                   response_model_counts=dict(Counter(r['response_model'] for r in rows if r['response_model'])),
                   generation_intervals=[{'source_run_protocol_id': p['identity']['protocol_id'],
                                         'first_attempt_utc': ls['entries'][0].get('started_at_utc') if ls['entries'] else None,
                                         'last_attempt_utc': ls['entries'][-1].get('finished_at_utc') if ls['entries'] else None}
                                        for p, ls in ((parent_plan, parent_ledger), (plan, ledger))],
                   interpretation={'P': 'GT attributes only; layout not specified', 'E': 'complete discrete GT specification including internal/world coordinates; E-O is not pure language information loss',
                                   'J': 'information-preserving JSON reserialization, not legacy Structured IR',
                                   'reference_cap_exceeders': 'Retained in all primary planned denominators; no cap-driven exclusions'},
                   research_api_requests_sent_by_aggregation=0)
    return {'version': 'fixed960_join_v1', 'manifest_identity_hash': manifest['identity_hash'],
            'semantic_protocol_id': semantic, 'rows': rows, 'summary': summary}


def collect(plan_path, destination):
    from . import plan as planning, runner
    plan = planning.load_plan(plan_path); identity = plan['identity']
    directory = runner._checked_path(identity['enabled_directory'])
    # Historical parent verification uses immutable hashes, never UTC freshness.
    with runner._parent_locked(identity):
        parent_plan = read_json(identity['parent']['plan_path'])
        parent_dir = Path(identity['parent']['enabled_directory'])
        def join(ledger):
            return aggregate_verified(plan, parent_plan, read_json(parent_dir / 'evaluation.json'),
                                      read_json(directory / 'evaluation.json') if (directory / 'evaluation.json').exists() else None,
                                      read_json(parent_dir / 'ledger.json'), ledger)
        if directory.exists():
            with runner.run_lock(directory), runner._authority_locked(identity):
                _, ledger = runner._load_enabled(plan)
                result = join(ledger)
        else:
            result = join({'entries': [], 'halt_reason': None})
    destination = runner._checked_path(destination)
    if destination.exists():
        raise GenerationError('new_aggregate_destination_required')
    destination.mkdir(parents=True)
    write_json(destination / 'aggregate.json', result)
    fields = ('provider', 'scene_id', 'condition', 'description_model', 'input_id', 'schedule_index', 'status', 'attempt_id', 'failure_kind', 'response_model', 'started_at_utc', 'finished_at_utc')
    with (destination / 'rows.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in result['rows'])
    return result['summary']
