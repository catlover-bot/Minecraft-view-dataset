"""CPU-only Goal015 saved-response evaluation and paired pilot summaries.

Imports no provider, API transport, credential, launcher or generation module.
Normal empty programs remain protocol-valid. Geometry failures and undefined
success scalars are never converted into model-failure utility zeroes.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import ast
import datetime as dt
import fcntl
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from ...server_v2 import core
from ...server_v2.alignment import AlignmentResourceLimit
from ...server_v2 import alignment
from ...analysis.goal010v2 import metrics as extended

BUILDERS = ('openai', 'anthropic')
CONDITIONS = ('SUMMARY_ONLY', 'FULL_RECORD')
POLICIES = ('completed_attempt_utility', 'common_valid_conditional_accuracy')
PRIMARY_METRICS = ('aligned_occupancy_iou', 'aligned_surface_f1_at_1')
DIAGNOSTIC_METRICS = ('aligned_occupancy_precision', 'aligned_occupancy_recall',
                      'aligned_dimension_mae', 'aligned_material_aware_iou')
METRICS = PRIMARY_METRICS + DIAGNOSTIC_METRICS
MODEL_FAILURES = frozenset(('parse_failure', 'validation_failure', 'execution_failure', 'truncation', 'refusal'))
IDENTITY = ('call_id', 'scene_id', 'builder', 'condition', 'repeat_index', 'paired_block_id')
SEED, DRAWS, MIN_FINITE = 20261017, 2000, 1900


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@lru_cache(maxsize=1)
def _frozen_object_hash():
    """Load only the three frozen pure hash helpers, without provider imports.

    Goal005/continuation GT descriptors use generation.object_hash. Importing
    that module would import the research transport module, so preserve these
    exact AST function bodies in a namespace containing only json/hashlib.
    """
    path = Path(core.__file__).with_name('generation.py')
    names = {'json_bytes', 'digest', 'object_hash'}
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in functions} == names and len(functions) == 3, 'frozen GT hash helper definitions differ')
    namespace = {'json': json, 'hashlib': hashlib}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['object_hash']


def frozen_grid_sha256(grid):
    return _frozen_object_hash()(core.grid_to_payload(grid))


def evaluate_grids(gt, prediction, *, limits=core.Limits()):
    """Reuse strict occupancy alignment once, and that shift for every metric."""
    strict = core.evaluate_pair(gt, prediction, limits=limits, alignment_mode='strict_translation')
    require(strict['metric_version'] == 'server_v2.2_strict_translation', 'frozen strict evaluator required')
    shift = strict['alignment']['shift_xyz']
    original = strict['aligned']
    counts = (original['occupancy_intersection'], original['pred_voxel_count'], original['gt_voxel_count'])
    values = {
        'aligned_occupancy_iou': original['occupancy_iou'],
        'aligned_surface_f1_at_1': None,
        'aligned_occupancy_precision': counts[0]/counts[1] if counts[1] else None,
        'aligned_occupancy_recall': counts[0]/counts[2] if counts[2] else None,
        'aligned_dimension_mae': original['dimension_mae'],
        'aligned_material_aware_iou': original['material_aware_iou'],
    }
    nulls = {name: 'EMPTY_PREDICTION' if name.endswith('precision') else 'EMPTY_GT'
             for name, value in values.items() if value is None and name != 'aligned_surface_f1_at_1'}
    try:
        extra = extended.compute_metrics(gt.as_world().voxels, prediction.as_world().voxels, shift)
    except (extended.MetricResourceLimit, MemoryError) as exc:
        nulls['aligned_surface_f1_at_1'] = 'METRIC_RESOURCE_LIMIT:' + getattr(exc, 'reason', 'MEMORY_ERROR')
        provenance = {'alignment_searched': False, 'stored_alignment_shift_xyz': list(shift),
                      'implementation_sha256': extended.implementation_sha256()}
    else:
        for name in ('occupancy_iou', 'material_aware_iou'):
            require(extra['aligned'][name] == original[name], 'frozen core/extended scalar parity differs')
        for name in ('occupancy_precision', 'occupancy_recall', 'surface_f1_at_1'):
            values['aligned_'+name] = extra['aligned'][name]
            if extra['aligned'][name] is None:
                nulls['aligned_'+name] = extra['details']['aligned']['null_reasons'][name]
        provenance = extra['provenance']
        require(provenance['stored_alignment_shift_xyz'] == list(shift)
                and provenance['alignment_searched'] is False, 'all metrics must share occupancy-selected shift')
    return {'metric_version': strict['metric_version'], 'values': values, 'null_reasons': nulls,
            'alignment': strict['alignment'], 'extended_provenance': provenance,
            'gt_grid_sha256': frozen_grid_sha256(gt),
            'prediction_grid_sha256': frozen_grid_sha256(prediction),
            'frozen_implementation_sha256': {
                'server_v2_core': _sha(Path(core.__file__).read_bytes()),
                'server_v2_alignment': _sha(Path(alignment.__file__).read_bytes()),
                'original_gt_hash_helpers': _sha(Path(core.__file__).with_name('generation.py').read_bytes()),
                'goal010v2_metrics': extended.implementation_sha256()},
            'requested_metrics_only': list(METRICS), 'failure0_applied': False}


def _base(record, observation):
    row = {key: record[key] for key in IDENTITY}
    row.update(state='NOT_SENT', sent=False, send_started=False, send_status='NOT_SENT',
               response_received=False, completed_response=False,
               usage_known=False, model_output_failure=False, failure_kind=None,
               protocol_valid=False, nonempty=False, empty_output=None,
               status='NOT_SENT', evaluation_status='NOT_COMPUTED', null_reason='NOT_SENT',
               metrics=None, artifacts={}, parsed_document=None, voxel_payload=None)
    if observation is not None:
        require(all(observation.get(key) == record[key] for key in IDENTITY), 'saved call/schedule identity differs')
        if record.get('payload') is not None:
            require(observation.get('payload') == record['payload'], 'saved call/request payload identity differs')
        for name in ('state', 'sent', 'send_status', 'response_received', 'completed_response', 'usage_known',
                     'model_output_failure', 'failure_kind', 'response_model', 'http_status'):
            if name in observation:
                row[name] = observation[name]
        if 'send_status' not in observation:
            row['send_status'] = ('POST_RETURNED' if row['response_received'] else
                                  'POSSIBLY_SENT' if row['sent'] else 'NOT_SENT')
        require(row['send_status'] in ('NOT_SENT', 'POSSIBLY_SENT', 'POST_RETURNED'), 'unknown send evidence')
        row['send_started'] = row['send_status'] != 'NOT_SENT'
        row['sent'] = row['send_status'] == 'POST_RETURNED' or row['response_received']
        for name in ('sent', 'response_received', 'completed_response', 'usage_known', 'model_output_failure'):
            require(type(row[name]) is bool, 'explicit boolean response evidence required: '+name)
        row['artifacts'] = {name: observation[name] for name in ('raw', 'response_text_path') if observation.get(name)}
    row['selected_call_id'] = row['call_id']
    if record.get('payload'):
        row['artifacts']['request_payload'] = record['payload']
    return row


def failure0_eligible(row):
    return (row['state'] == 'SETTLED' and row['sent'] is True and row['response_received'] is True
            and row['usage_known'] is True and row['model_output_failure'] is True
            and row['failure_kind'] in MODEL_FAILURES and row['protocol_valid'] is False)


def evaluate_one(record, observation, gt=None, *, limits=core.Limits()):
    """Evaluate one saved response. ``gt`` can be a Grid or lazy zero-arg loader.

    The filesystem wrapper authenticates descriptors first. The decoded text in
    observation['response_text'] is transient and never part of a public table.
    A lazy GT loader is called only after output protocol validation succeeds.
    """
    row = _base(record, observation)
    if observation is None or row['state'] == 'NOT_SENT':
        return row
    if row['state'] != 'SETTLED' or not row['response_received'] or not row['usage_known']:
        row.update(status='OPERATIONAL_UNKNOWN', null_reason=row['failure_kind'] or row['state'])
        return row
    if row['failure_kind']:
        row.update(status='MODEL_OUTPUT_FAILURE' if failure0_eligible(row) else 'OPERATIONAL_FAILURE',
                   null_reason=row['failure_kind'])
        return row
    if not row['completed_response']:
        row.update(status='OPERATIONAL_UNKNOWN', null_reason='RESPONSE_COMPLETION_NOT_CONFIRMED')
        return row
    phase = 'parse_failure'
    try:
        document = core.strict_json(observation.get('response_text'), limits)
        phase = 'validation_failure'
        require(isinstance(document, dict) and set(document) == {'operations'}, 'native action object required')
        actions = core.validate_actions(document['operations'], limits)
        phase = 'execution_failure'
        prediction = core.execute(actions, limits)
    except (core.ValidationError, ValueError, TypeError):
        row.update(status='MODEL_OUTPUT_FAILURE', null_reason=phase, failure_kind=phase,
                   model_output_failure=True)
        return row
    except MemoryError:
        row.update(status='LOCAL_RESOURCE_FAILURE', null_reason='LOCAL_EXECUTION_MEMORY_LIMIT',
                   evaluation_status='LOCAL_EXECUTION_MEMORY_LIMIT')
        return row
    row.update(status='PROTOCOL_VALID', null_reason=None, protocol_valid=True,
               nonempty=bool(prediction.voxels), empty_output=not bool(prediction.voxels),
               parsed_document=document, voxel_payload=core.grid_to_payload(prediction),
               operation_count=len(actions), final_voxel_count=len(prediction.voxels))
    if gt is None:
        row.update(evaluation_status='NOT_COMPUTED', null_reason='GT_NOT_AVAILABLE')
        return row
    try:
        truth = gt() if callable(gt) else gt
        require(isinstance(truth, core.Grid), 'evaluation GT must use the frozen Grid reader')
        row['metrics'] = evaluate_grids(truth, prediction, limits=limits)
        row['evaluation_status'] = 'PASS'
    except AlignmentResourceLimit as exc:
        row.update(evaluation_status='ALIGNMENT_RESOURCE_LIMIT', null_reason='ALIGNMENT_RESOURCE_LIMIT',
                   evaluation_failure_metadata=exc.metadata)
    except (extended.MetricResourceLimit, MemoryError, OSError, core.ValidationError) as exc:
        row.update(evaluation_status='EVALUATION_UNAVAILABLE', null_reason=type(exc).__name__)
    return row


def _schedule(schedule, fixture):
    require(isinstance(schedule, list) and schedule, 'planned schedule required')
    scenes = sorted({row['scene_id'] for row in schedule})
    require(fixture or len(scenes) == 10, 'production requires ten planned scenes')
    keys = [(r['scene_id'], r['builder'], r['condition'], r['repeat_index']) for r in schedule]
    expected = {(scene, builder, condition, repeat) for scene in scenes for builder in BUILDERS
                for condition in CONDITIONS for repeat in (1, 2)}
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'complete fixed scene/Builder/condition/repeat frame required')
    require(len({r['call_id'] for r in schedule}) == len(schedule), 'unique selected call IDs required')
    blocks = {}
    for row in schedule:
        require(type(row['repeat_index']) is int, 'repeat index must be an integer')
        blocks.setdefault(row['paired_block_id'], []).append(row)
    require(len(blocks) == len(scenes)*4 and all(len(rows) == 2 for rows in blocks.values()), 'two calls per paired block required')
    for rows in blocks.values():
        require(len({(r['scene_id'], r['builder'], r['repeat_index']) for r in rows}) == 1
                and {r['condition'] for r in rows} == set(CONDITIONS), 'paired block identity differs')
    return scenes


def _value(row, metric, policy):
    if row['protocol_valid']:
        if row['metrics'] is None:
            return None, row['null_reason'] or 'METRICS_UNAVAILABLE', False
        value = row['metrics']['values'][metric]
        if value is None:
            return None, row['metrics']['null_reasons'].get(metric, 'UNDEFINED_SUCCESS_SCALAR'), False
        require(_finite(value) and (metric == 'aligned_dimension_mae' or 0 <= value <= 1), 'invalid metric scalar')
        return value, None, False
    if policy == 'completed_attempt_utility' and metric in PRIMARY_METRICS and failure0_eligible(row):
        return 0., None, True
    return None, row['null_reason'] or 'PROTOCOL_NOT_VALID', False


def _interval(point, values, buildings):
    finite = [value for value in values if value is not None]
    require(len(values) == DRAWS and all(_finite(v) for v in finite), 'exactly 2000 finite-or-null draws required')
    passed = point is not None and buildings >= 2 and len(finite) >= MIN_FINITE
    return {'ci95': [float(v) for v in np.quantile(finite, [.025, .975], method='linear')] if passed else None,
            'ci_status': 'EXPLORATORY_CONDITIONAL_ON_FINITE_DRAWS' if passed else 'INSUFFICIENT_BOOTSTRAP_VALIDITY',
            'bootstrap_finite_draws': len(finite), 'bootstrap_undefined_draws': DRAWS-len(finite),
            'bootstrap_draws': values}


def _paired(rows, scenes, builder, metric, policy, draws):
    indexed = {(r['scene_id'], r['builder'], r['condition'], r['repeat_index']): r for r in rows}
    buildings, individual = [], []
    for scene in scenes:
        differences = []
        for repeat in (1, 2):
            left = indexed[scene, builder, 'SUMMARY_ONLY', repeat]
            right = indexed[scene, builder, 'FULL_RECORD', repeat]
            a, reason_a, zero_a = _value(left, metric, policy)
            b, reason_b, zero_b = _value(right, metric, policy)
            difference = b-a if a is not None and b is not None else None
            if difference is not None:
                differences.append(difference)
            individual.append({'scene_id': scene, 'repeat_index': repeat,
                'paired_block_id': left['paired_block_id'], 'summary_call_id': left['call_id'], 'full_call_id': right['call_id'],
                'summary_value': a, 'full_value': b, 'difference_full_minus_summary': difference,
                'summary_null_reason': reason_a, 'full_null_reason': reason_b,
                'summary_failure0': zero_a, 'full_failure0': zero_b})
        buildings.append({'scene_id': scene, 'eligible_repeats': len(differences),
                          'mean_paired_difference': mean(differences) if differences else None})
    values = [r['mean_paired_difference'] for r in buildings if r['eligible_repeats']]
    point = mean(values) if values else None
    sampled = []
    for draw in draws:
        current = [buildings[int(index)]['mean_paired_difference'] for index in draw
                   if buildings[int(index)]['mean_paired_difference'] is not None]
        sampled.append(mean(current) if current else None)
    return {'builder': builder, 'metric': metric, 'estimand': policy,
            'contrast': 'FULL_RECORD_MINUS_SUMMARY_ONLY', 'estimate': point,
            'estimate_status': 'DEFINED' if point is not None else 'NO_ELIGIBLE_PAIRS',
            'planned_scenes': len(scenes), 'planned_paired_blocks': len(scenes)*2,
            'eligible_paired_blocks': sum(r['difference_full_minus_summary'] is not None for r in individual),
            'contributing_scenes': len(values), 'scenes_with_both_repeats': sum(r['eligible_repeats']==2 for r in buildings),
            'scenes_with_one_repeat': sum(r['eligible_repeats']==1 for r in buildings),
            'scenes_with_no_repeats': sum(r['eligible_repeats']==0 for r in buildings),
            'failure0_condition_values': sum(r['summary_failure0']+r['full_failure0'] for r in individual),
            'individual_paired_differences': individual, 'per_scene': buildings,
            'weighting': 'available paired-repeat differences averaged within scene; scenes equally weighted',
            **_interval(point, sampled, len(values))}


def aggregate_rows(schedule, evaluated_rows, *, fixture=False):
    """Preserve 80 planned rows, 40 S/F blocks and independent N at most ten."""
    scenes = _schedule(schedule, fixture)
    indexed = {row['call_id']: row for row in evaluated_rows}
    require(len(indexed) == len(evaluated_rows), 'duplicate evaluated selected call')
    require(set(indexed) <= {r['call_id'] for r in schedule}, 'unplanned evaluated call')
    rows = []
    for record in schedule:
        row = indexed.get(record['call_id'], _base(record, None))
        require(all(row[k] == record[k] for k in IDENTITY), 'evaluation identity differs from planned call')
        rows.append(row)
    draws = np.random.Generator(np.random.PCG64(SEED)).integers(0, len(scenes), size=(DRAWS, len(scenes)), dtype=np.int64)
    results = [_paired(rows, scenes, builder, metric, policy, draws)
               for builder in BUILDERS for metric in PRIMARY_METRICS for policy in POLICIES]
    reliability, diagnostics = [], []
    for builder in BUILDERS:
        for condition in CONDITIONS:
            group = [r for r in rows if r['builder']==builder and r['condition']==condition]
            count = len(group)
            item = {'builder': builder, 'condition': condition, 'planned_calls': count,
                    'planned_scenes': len(scenes),
                    'status_counts': dict(Counter(r['status'] for r in group)),
                    'failure_kind_counts': dict(Counter(r['failure_kind'] for r in group if r['failure_kind']))}
            for name in ('send_started', 'sent', 'response_received', 'completed_response', 'protocol_valid', 'nonempty'):
                item[name+'_count'] = sum(r[name] is True for r in group)
                item[name+'_rate_over_planned'] = item[name+'_count']/count
            item['normal_empty_protocol_valid_count'] = sum(r['protocol_valid'] and r['empty_output'] is True for r in group)
            item['usage_unknown_sent_count'] = sum(r['sent'] and not r['usage_known'] for r in group)
            item['possibly_sent_unconfirmed_count'] = sum(r['send_status']=='POSSIBLY_SENT' and not r['response_received'] for r in group)
            item['sent_count_definition'] = 'POST returned or HTTP response received; uncertain send starts counted separately'
            reliability.append(item)
            for metric in DIAGNOSTIC_METRICS:
                by_scene = []
                for scene in scenes:
                    values = [_value(r, metric, 'common_valid_conditional_accuracy')[0]
                              for r in group if r['scene_id']==scene]
                    valid = [v for v in values if v is not None]
                    by_scene.append({'scene_id': scene, 'valid_rows': len(valid), 'mean': mean(valid) if valid else None})
                valid_scenes = [r['mean'] for r in by_scene if r['mean'] is not None]
                diagnostics.append({'builder': builder, 'condition': condition, 'metric': metric,
                    'description_only': True, 'failure0_applied': False, 'planned_calls': count,
                    'valid_rows': sum(r['valid_rows'] for r in by_scene), 'contributing_scenes': len(valid_scenes),
                    'equal_scene_mean': mean(valid_scenes) if valid_scenes else None, 'per_scene': by_scene})
    return {'schema': 'goal015_paired_pilot_summary_v1', 'research_status': 'DEVELOPMENT_PILOT_EXPLORATORY',
            'planned_calls': len(schedule), 'planned_paired_blocks': len(schedule)//2,
            'planned_independent_buildings': len(scenes), 'planned_scene_ids': scenes,
            'evaluated_protocol_valid_rows': sum(r['protocol_valid'] for r in rows),
            'actual_metric_rows': sum(r['metrics'] is not None for r in rows),
            'primary_paired_estimates': results, 'reliability': reliability, 'diagnostic_descriptives': diagnostics,
            'inference': {'rng': 'NumPy Generator(PCG64)', 'seed': SEED, 'replicates': DRAWS,
                'index_matrix_sha256_little_endian_int64': _sha(draws.astype('<i8').tobytes()),
                'planned_scene_frame_retained': True, 'same_draws_for_every_paired_estimand': True,
                'minimum_finite_draws': MIN_FINITE, 'minimum_contributing_scenes': 2,
                'quantile_method': 'linear / Hyndman-Fan type 7', 'p_values_computed': False},
            'interpretation': {'utility_unconditional_effect_if_unknown_missing': False,
                'common_valid_has_treatment_dependent_selection': True,
                'replicates_independent_buildings': False, 'power_guaranteed': False},
            'research_api_requests_sent_by_evaluation': 0}


def _read_descriptor(desc):
    path = Path(desc['path'])
    require(path.is_absolute() and '..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            'absolute nonsymlink saved artifact required')
    raw = path.read_bytes()
    require(len(raw)==desc['bytes'] and _sha(raw)==desc['sha256'], 'saved response artifact hash mismatch')
    return raw


def _write_new(path, value):
    raw = _json_bytes(value)
    with path.open('xb') as stream:
        stream.write(raw)
    return {'path': str(path), 'sha256': _sha(raw), 'bytes': len(raw)}


def evaluate_saved(schedule, saved_calls, output_dir, gt_loader, *, fixture=False,
                   plan_binding=None, gt_sources=None):
    """Verify saved response descriptors, evaluate, and exclusively save products.

    gt_loader(scene_id) must read only the plan-bound original GT descriptors.
    An absent ledger is supplied as saved_calls=[]; all planned NULLs survive.
    This function neither creates authority/approval nor imports the runtime.
    """
    _schedule(schedule, fixture)
    directory = Path(output_dir).absolute()
    require('..' not in directory.parts and not any(p.is_symlink() for p in (directory, *directory.parents)), 'safe evaluation output required')
    if not fixture:
        relative = directory.relative_to('/root/minecraft-runs')
        require(len(relative.parts)>=2 and relative.parts[0].startswith('goal015-'), 'new Goal015 output required')
    directory.mkdir(parents=True, exist_ok=False)
    indexed = {r['call_id']:r for r in saved_calls}
    require(len(indexed)==len(saved_calls) and set(indexed)<={r['call_id'] for r in schedule}, 'saved call frame differs')
    evaluated = []
    for index, record in enumerate(schedule):
        if record.get('payload'):
            _read_descriptor(record['payload'])
        observation = indexed.get(record['call_id'])
        if observation is not None:
            observation = dict(observation)
            if observation.get('response_received') is True:
                require(observation.get('raw') is not None, 'received response requires saved raw bytes')
            if observation.get('raw'):
                _read_descriptor(observation['raw'])
            if observation.get('response_text_path'):
                observation['response_text'] = _read_descriptor(observation['response_text_path']).decode('utf-8')
            elif observation.get('state') == 'SETTLED' and not observation.get('failure_kind'):
                require(False, 'settled response text artifact is missing')
        row = evaluate_one(record, observation, lambda: gt_loader(record['scene_id']))
        if gt_sources is not None:
            row['gt_source'] = gt_sources[record['scene_id']]
        product_dir = directory/f'{index:03d}'
        product_dir.mkdir()
        for field, name in (('parsed_document', 'parsed.json'), ('voxel_payload', 'voxels.json')):
            value = row.pop(field)
            if value is not None:
                row['artifacts'][name] = _write_new(product_dir/name, value)
        if row['metrics'] is not None:
            row['artifacts']['metrics.json'] = _write_new(product_dir/'metrics.json', row['metrics'])
        evaluated.append(row)
    summary = aggregate_rows(schedule, evaluated, fixture=fixture)
    summary['plan_binding'] = plan_binding
    summary['schedule_sha256'] = _sha(_json_bytes(schedule))
    summary['evaluation_artifact'] = _write_new(directory/'evaluation_80.json', {
        'rows': evaluated, 'planned_calls': len(schedule), 'research_api_requests_sent': 0,
        'plan_binding': plan_binding, 'schedule_sha256': summary['schedule_sha256']})
    _write_new(directory/'summary.json', summary)
    return summary


def evaluate_plan(plan_path, output_dir=None):
    """Offline CLI entry: immutable plan plus its existing canonical ledger.

    The runtime's read-only verifier imports no adapter. No execution lock,
    approval, authority or enabled directory is created. GT bytes are loaded only
    for protocol-valid saved responses and checked against original descriptors.
    """
    from . import runtime
    from ...server_v2.data import load_scene
    plan = runtime.verify_plan(plan_path)
    root = Path(plan['canonical_root'])
    target = Path(output_dir).absolute() if output_dir else root/'evaluations'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    require(target.is_relative_to(root) and target != root, 'evaluation output must stay inside this canonical plan root')
    with ExitStack() as stack:
        lock_path = root/'execution.lock'
        if lock_path.exists():
            stream = stack.enter_context(runtime.path_safe(lock_path).open('rb'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('generation is active; evaluate only saved stable responses') from None
            stack.callback(fcntl.flock, stream.fileno(), fcntl.LOCK_UN)
        ledger_path = root/'enabled/ledger.json'
        calls = []
        if ledger_path.exists():
            require(lock_path.exists(), 'canonical ledger requires an existing execution lock')
            ledger = runtime.read_json(ledger_path)
            approval = runtime.read_json(root/'enabled/approval.json')
            runtime._validate_ledger(plan, ledger, approval)
            calls = ledger['calls']
        sources = plan.get('gt_sources', {})
        require(set(sources) == {r['scene_id'] for r in plan['schedule']}, 'all planned GT source descriptors required')
        grids = {}
        def load_gt(scene):
            if scene not in grids:
                source = sources[scene]
                building = runtime.path_safe(source['building_directory'])
                for relative, field in (('gt/voxels.npy', 'voxels_sha256'), ('gt/bbox.json', 'bbox_sha256')):
                    require(runtime.descriptor(building/relative)['sha256'] == source[field], 'original GT bytes differ')
                loaded = load_scene(building)
                require(loaded.scene_id == scene, 'GT scene identity differs')
                require(frozen_grid_sha256(loaded.grid) == source['grid_sha256'], 'original GT normalized grid differs')
                grids[scene] = loaded.grid
            return grids[scene]
        binding = {'plan_id': plan['plan_id'], 'plan': runtime.descriptor(plan_path),
                   'ledger': runtime.descriptor(ledger_path) if ledger_path.exists() else None}
        return evaluate_saved(plan['schedule'], calls, target, load_gt,
            fixture=plan.get('fixture_only') is True, plan_binding=binding, gt_sources=sources)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Goal015 saved-response CPU evaluation and paired aggregation; no research API')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summary = evaluate_plan(args.plan, args.output_dir)
    print(json.dumps({'planned_calls': summary['planned_calls'], 'actual_metric_rows': summary['actual_metric_rows'],
                      'evaluation_artifact': summary['evaluation_artifact'], 'research_api_requests_sent': 0}))


if __name__ == '__main__':
    main()
