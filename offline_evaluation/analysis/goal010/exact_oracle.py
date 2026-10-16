"""Post-hoc CPU audit of every frozen E outcome and its exact-spec control.

The caller holds the shared read-only source locks and verifies the committed
post-hoc specification. This module performs no alignment search, generation,
tokenization, network access, credential lookup or original-file writes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time

from ...resmoke_v2 import cuboids
from ...server_v2.core import Grid, Limits, execute, payload_to_grid, strict_json, validate_actions
from ...server_v2.data import load_scene, sha256_file

VERSION = 'goal010_exact_oracle_audit_v1'
ANALYSIS_LABEL = 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC'
METRIC_VERSION = 'server_v2.2_strict_translation'
BUILDERS = ('openai', 'anthropic')
NEAR_PERFECT_OCCUPANCY = .99
LOW_EXACT_MATERIAL_IOU = .5
ALIAS_SOURCE = Path(__file__).resolve().parents[3] / 'tools/evaluate_rebuild_metrics.py'
ALIAS_SOURCE_SHA256 = 'd67bd07007c9db56c532859763a41dbc26ced3d9c00e0905c24d38e6c5fb807c'
LEXICAL_ALIASES = {'stone_bricks': 'stonebrick', 'stone_brick': 'stonebrick', 'stonebrick': 'stonebrick',
                   'minecraft_stone_bricks': 'stonebrick', 'minecraft_stone_brick': 'stonebrick',
                   'brick_block': 'brick', 'bricks': 'brick'}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def object_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def regular(path):
    path = Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'regular nonsymlink source required')
    return path


def read_json(path):
    return json.loads(regular(path).read_text(encoding='utf-8'))


def check_gate(gate, specification_sha256=None):
    require(isinstance(gate, dict) and gate.get('status') == 'PASS'
            and gate.get('input_kind') in ('fixture', 'verified_sources')
            and gate.get('phase0_binding') == 'PASS' and gate.get('definitions_frozen') is True,
            'Phase0 binding and frozen post-hoc specification required before E analysis')
    for key, length in (('specification_sha256', 64), ('goal007_manifest_sha256', 64), ('code_commit', 40)):
        require(isinstance(gate.get(key), str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', gate[key]),
                'explicit frozen gate hash required: ' + key)
    if specification_sha256 is not None:
        require(gate['specification_sha256'] == specification_sha256, 'frozen specification SHA mismatch')


def _checked(path, digest, inventory, size=None):
    path = regular(path)
    actual = sha256_file(path)
    require(actual == digest and (size is None or path.stat().st_size == size), 'source hash/size mismatch: ' + str(path))
    inventory[str(path)] = actual
    return path


def _artifact(root, descriptor, inventory):
    relative = Path(descriptor['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'artifact traversal rejected')
    path = regular(Path(root) / relative)
    require(path.resolve().is_relative_to(Path(root).resolve()), 'artifact escaped canonical run')
    return _checked(path, descriptor['sha256'], inventory, descriptor['bytes'])


def bind_sources(slots, goal007_manifest_path, *, gate):
    """Bind selected E attempts using only canonical references in the receipt."""
    check_gate(gate)
    inventory = {}
    path = _checked(goal007_manifest_path, gate['goal007_manifest_sha256'], inventory)
    manifest = read_json(path)
    descriptor = manifest['products']['data/final_slots_960.json']
    slot_path = _checked(path.parent / 'data/final_slots_960.json', descriptor['sha256'], inventory, descriptor['bytes'])
    require(object_hash(slots) == object_hash(read_json(slot_path)), 'selected slots differ from frozen Goal007 product')
    source_manifest = manifest['source_manifest']
    fixed_path = _checked(source_manifest['path'], source_manifest['sha256'], inventory)
    fixed = read_json(fixed_path)
    require(fixed['identity_hash'] == object_hash(fixed['identity']) == source_manifest['identity_hash'], 'fixed schedule identity mismatch')
    full = {r['schedule_index']: r for r in fixed['identity']['full_schedule']}
    selected = [r for r in slots if r['condition'] == 'E']
    require(len({(r['scene_id'], r['provider']) for r in selected}) == len(selected), 'duplicate selected E outcome')
    scenes = {r['scene_id'] for r in selected}
    require({(r['scene_id'], r['provider']) for r in selected} == {(scene, b) for scene in scenes for b in BUILDERS},
            'each E scene needs both Builder outcomes')
    if gate['input_kind'] == 'verified_sources':
        require(len(slots) == 960 and len(selected) == 120 and len(scenes) == 60, 'fixed960/E120/scene60 scope required')
        require(set(manifest['sources']) == {'parent160', 'original781', 'supplement3', 'supplement17'}, 'all four canonical sources required')
    sources = {}
    for name, reference in manifest['sources'].items():
        plan_path = _checked(reference['plan_path'], reference['plan_sha256'], inventory)
        ledger_path = _checked(reference['ledger_path'], reference['ledger_sha256'], inventory)
        plan, ledger = read_json(plan_path), read_json(ledger_path)
        require(plan['identity_hash'] == object_hash(plan['identity']) == reference['plan_identity_hash']
                and ledger['plan_identity_hash'] == plan['identity_hash'], 'source plan/ledger identity mismatch')
        require(Path(plan['identity']['enabled_directory']).absolute() == Path(reference['enabled_directory']).absolute()
                and ledger_path == Path(reference['enabled_directory']).absolute() / 'ledger.json', 'canonical source directory mismatch')
        sources[name] = (reference, plan, ledger)
    packages = []
    for row in sorted(selected, key=lambda r: (r['scene_id'], r['provider'])):
        reference, plan, ledger = sources[row['source_label']]
        records = [r for r in plan['identity']['schedule'] if r['schedule_index'] == row['global_schedule_index']]
        entries = [r for r in ledger['entries'] if r['attempt_id'] == row['attempt_id']]
        require(len(records) == len(entries) == 1, 'selected E attempt not unique in original source')
        record, entry = records[0], entries[0]
        require(record == full[row['global_schedule_index']], 'source E input differs from fixed full schedule')
        require(Path(row['source_run']).absolute() == Path(reference['enabled_directory']).absolute()
                and Path(row['source_plan_path']).absolute() == Path(reference['plan_path']).absolute()
                and row['source_plan_identity'] == plan['identity_hash'] and row['source_entry_hash'] == object_hash(entry),
                'selected E original source binding mismatch')
        for key in ('provider', 'input_id', 'scene_id', 'condition'):
            require(record[key] == row[key], 'selected E record mismatch: ' + key)
        require(entry['schedule_index'] == row['global_schedule_index'] and entry['input_id'] == row['input_id']
                and entry['provider'] == row['provider'] and entry['status'] == row['generation_status']
                and entry.get('usage') == row['usage'] and entry['artifacts'] == row['artifacts'], 'selected E ledger mismatch')
        require(hashlib.sha256(record['input_text'].encode()).hexdigest() == record['input_sha256'] == row['input_sha256'], 'E input byte hash mismatch')
        gt = plan['identity']['gt_by_scene'][row['scene_id']]
        require(object_hash(gt) == row['source_gt_descriptor_hash'] and gt['grid_sha256'] == row['gt_grid_sha256'], 'E GT descriptor binding mismatch')
        directory = Path(gt['building_directory'])
        _checked(directory / 'gt/voxels.npy', gt['voxels_sha256'], inventory)
        _checked(directory / 'gt/bbox.json', gt['bbox_sha256'], inventory)
        artifacts = {name: str(_artifact(reference['enabled_directory'], descriptor, inventory)) for name, descriptor in entry['artifacts'].items()}
        packages.append({'slot': row, 'record': record, 'gt': gt, 'limits': plan['identity']['limits'],
                         'output_token_cap': plan['identity']['budgets']['max_output_tokens'], 'artifacts': artifacts})
    return packages, inventory


def _dimensions(voxels):
    return [max(c[i] for c in voxels) - min(c[i] for c in voxels) + 1 for i in range(3)] if voxels else [0, 0, 0]


def independent_metrics(gt_voxels, pred_voxels):
    """Independent exact-ID arithmetic on already positioned world dictionaries."""
    gt_count, pred_count = len(gt_voxels), len(pred_voxels)
    pairs = Counter()
    for coordinate, material in gt_voxels.items():
        pairs[material, pred_voxels.get(coordinate, 'air')] += 1
    for coordinate, material in pred_voxels.items():
        if coordinate not in gt_voxels:
            pairs['air', material] += 1
    matched = sum(count for (g, p), count in pairs.items() if g == p and g != 'air')
    add = sum(count for (g, p), count in pairs.items() if p == 'air')
    delete = sum(count for (g, p), count in pairs.items() if g == 'air')
    replace = sum(count for (g, p), count in pairs.items() if g != p and g != 'air' and p != 'air')
    overlap, union = matched + replace, sum(pairs.values())
    materials_g, materials_p = set(gt_voxels.values()), set(pred_voxels.values())
    material_count = len(materials_g) + len(materials_p)
    total, normalization = gt_count + pred_count, max(1, gt_count)
    gd, pd = _dimensions(gt_voxels), _dimensions(pred_voxels)
    result = {'gt_voxel_count': gt_count, 'pred_voxel_count': pred_count,
        'occupancy_intersection': overlap, 'occupancy_union': union,
        'occupancy_iou': overlap / union if union else 1.0, 'occupancy_f1': 2 * overlap / total if total else 1.0,
        'material_match_count': matched, 'material_aware_iou': matched / (total - matched) if total else 1.0,
        'material_set_f1': 2 * len(materials_g & materials_p) / material_count if material_count else 1.0,
        'dimension_mae': sum(abs(g - p) for g, p in zip(gd, pd)) / 3, 'gt_dimensions_xyz': gd, 'pred_dimensions_xyz': pd,
        'add_count': add, 'delete_count': delete, 'replace_count': replace,
        'add_normalized_cost': add / normalization, 'delete_normalized_cost': delete / normalization,
        'replace_normalized_cost': replace / normalization, 'total_normalized_repair_cost': (add + delete + replace) / normalization}
    require(gt_count == matched + replace + add and pred_count == matched + replace + delete, 'confusion mass imbalance')
    return result, pairs


def lexical_sensitivity(gt_voxels, pred_voxels):
    """A frozen historical lexical grouping, never an output repair or truth claim."""
    g = {c: LEXICAL_ALIASES.get(m, m) for c, m in gt_voxels.items()}
    p = {c: LEXICAL_ALIASES.get(m, m) for c, m in pred_voxels.items()}
    metrics, _ = independent_metrics(g, p)
    return {k: metrics[k] for k in ('material_aware_iou', 'material_set_f1', 'material_match_count')}


def verify_metrics(actual, expected):
    require(set(actual) == set(expected), 'strict metric field set mismatch')
    for name in actual:
        if isinstance(actual[name], float):
            require(type(expected[name]) in (int, float) and math.isfinite(expected[name])
                    and math.isclose(actual[name], expected[name], rel_tol=0, abs_tol=1e-12), 'strict metric disagreement: ' + name)
        else:
            require(type(actual[name]) is type(expected[name]) and actual[name] == expected[name]
                    and (not isinstance(actual[name], list) or all(type(v) is int for v in expected[name])),
                    'strict metric disagreement: ' + name)


def validate_alignment(metrics):
    require(metrics.get('metric_version') == METRIC_VERSION, 'strict translation metric version required')
    certificate = metrics.get('alignment', {})
    for field in ('search_complete', 'globally_optimal_unrestricted', 'all_aligned_metrics_share_shift'):
        require(certificate.get(field) is True, 'stored strict alignment certificate required')
    require(certificate.get('mode') == 'strict_translation' and all(certificate.get(k) is False for k in ('rotation', 'scale', 'clipping', 'resource_limit_reached')),
            'no approximate alignment, rotation, scaling or clipping permitted')
    shift = certificate.get('shift_xyz')
    require(isinstance(shift, list) and len(shift) == 3 and all(type(v) is int for v in shift), 'stored XYZ integer shift required')
    return shift


def _bounds(operation):
    if operation['op'] == 'set':
        return tuple(operation[a] for a in 'xyz') * 2
    return tuple(operation[a + suffix] for suffix in ('1', '2') for a in 'xyz')


def operation_transcription(spec, operations):
    """Describe literal boundary/label matches; equivalent repartitions may differ."""
    origin = spec['origin_xyz'] if spec['coordinate_frame'] == 'local' else [0, 0, 0]
    expected = {}
    for row in spec['cuboids']:
        bounds = tuple(row[i] + origin[i % 3] for i in range(6))
        expected[bounds] = spec['palette'][row[6]]
    counts = Counter(); labels = Counter(); exact_bounds = set(); correct_bounds = set(); substitutions = Counter()
    touches = 0
    for operation in operations:
        counts[operation['op']] += 1
        bounds = _bounds(operation)
        touches += math.prod(bounds[i + 3] - bounds[i] + 1 for i in range(3))
        material = operation.get('block', 'air'); labels[material] += 1
        if bounds in expected:
            exact_bounds.add(bounds)
            if material == expected[bounds]: correct_bounds.add(bounds)
            else: substitutions[expected[bounds], material] += 1
    return {'operations': len(operations), 'operation_kinds': dict(sorted(counts.items())), 'touched_voxel_visits': touches,
        'output_operation_labels': dict(sorted(labels.items())), 'labels_absent_from_E_palette': sorted(set(labels) - set(spec['palette']) - {'air'}),
        'reference_cuboids': len(expected), 'reference_boundaries_seen': len(exact_bounds),
        'reference_boundaries_with_correct_label': len(correct_bounds), 'reference_boundaries_without_literal_match': len(expected) - len(exact_bounds),
        'boundary_label_substitutions': [{'expected_material': a, 'operation_material': b, 'operation_count': n}
            for (a, b), n in sorted(substitutions.items())],
        'caveat': 'Literal cuboid-boundary diagnostic only; correct-label counts mean at least one matching operation, not necessarily the final write. Alternative valid partitions/order may produce identical voxels; no omission or causality is inferred from an unmatched boundary.'}


def reference_control(scene, input_text, limits):
    require('\n' in input_text, 'complete cuboid E input required')
    spec = strict_json(input_text.split('\n', 1)[1], limits)
    require(spec.get('schema') == cuboids.SEMANTICS['schema'], 'actual fixed cuboid E schema required, not historical Z-runs')
    encoded = cuboids.encode_exact(scene.grid, limits)
    require(encoded == spec and cuboids.exact_builder_text(encoded) == input_text, 'GT re-encoding differs from immutable E input')
    decoded = cuboids.decode_exact(spec, limits)
    require(decoded.to_dict() == scene.grid.to_dict(), 'GT/spec exact control failed')
    reference = cuboids.reference_actions(spec, limits)
    executed = execute(reference['operations'], limits, origin=scene.grid.origin)
    require(executed.to_dict() == scene.grid.as_world().to_dict(), 'reference execution is not exact GT')
    # A separate count of contiguous same-material +Z runs, not an old input rewrite.
    voxels = scene.grid.as_world().voxels
    z_runs = sum(voxels.get((x, y, z - 1)) != material for (x, y, z), material in voxels.items())
    control = {'scene_id': scene.scene_id, 'schema': spec['schema'], 'coordinate_frame': spec['coordinate_frame'],
        'origin_xyz': spec['origin_xyz'], 'storage_bbox': scene.bbox, 'occupied_dimensions_xyz': _dimensions(voxels),
        'storage_dimensions_xyz': [scene.bbox[a+'max']-scene.bbox[a+'min']+1 for a in 'xyz'],
        'gt_occupied_voxels': len(voxels), 'gt_grid_sha256': object_hash(scene.grid.to_dict()),
        'input_sha256': hashlib.sha256(input_text.encode()).hexdigest(), 'input_utf8_bytes': len(input_text.encode()),
        'input_characters': len(input_text), 'spec_utf8_bytes': len(cuboids.compact_json(spec).encode()),
        'palette': spec['palette'], 'palette_count': len(spec['palette']), 'cuboids': len(spec['cuboids']),
        'same_material_z_runs': z_runs, 'reference_operations': len(reference['operations']),
        'reference_compact_json_bytes': len(cuboids.compact_json(reference).encode()), 'reference_actions_sha256': object_hash(reference),
        'decoded_grid_sha256': object_hash(decoded.to_dict()), 'reference_executed_grid_sha256': object_hash(executed.to_dict()),
        'GT_spec_exact': True, 'reference_execution_exact': True, 'control_is_model_success': False,
        'reference_tokens': None, 'reference_token_capacity_status': 'NOT_CERTIFIED_NO_LOCAL_TOKENIZATION',
        'limits': asdict(limits), 'partition_is_proven_minimum': False, 'model_output_fit_guarantee': False}
    return spec, control


def audit_outcome(package, scene, spec, control):
    row = package['slot']; limits = Limits(**package['limits'])
    base = {key: row.get(key) for key in ('scene_id', 'provider', 'builder_model', 'response_model', 'global_schedule_index', 'attempt_id',
        'input_id', 'input_sha256', 'source_label', 'source_plan_path', 'source_plan_identity', 'source_entry_hash', 'source_gt_descriptor_hash',
        'gt_grid_sha256', 'status', 'generation_status', 'failure_kind', 'completed_request_outcome', 'usage_known')}
    base.update(analysis_label=ANALYSIS_LABEL, artifact_sha256={k:v['sha256'] for k,v in row['artifacts'].items()},
        GT_source=package['gt'], independent_metrics=None, lexical_sensitivity=None, operation_transcription=None,
        near_perfect_occupancy_low_material=None, same_material_set_with_spatial_replacements=None)
    usage = row.get('usage'); cap = package['output_token_cap']
    require(type(cap) is int and cap > 0, 'positive output cap from original plan required')
    require(type(row['usage_known']) is bool and type(row['completed_request_outcome']) is bool,
            'explicit known-usage/completion booleans required')
    require((usage is not None) == row['usage_known'], 'usage status and observed usage disagree')
    if usage is not None:
        require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens')),
                'nonnegative observed integer token counts required')
    output_tokens = usage.get('output_tokens') if usage else None
    reasoning_tokens = (usage.get('output_tokens_details') or {}).get('reasoning_tokens') if usage else None
    capacity = {**{k:base[k] for k in ('scene_id','provider','global_schedule_index','attempt_id','status','failure_kind','input_id','input_sha256')},
        **{k:control[k] for k in ('input_utf8_bytes','input_characters','spec_utf8_bytes','gt_occupied_voxels','cuboids',
            'same_material_z_runs','palette_count','reference_operations','reference_compact_json_bytes','reference_tokens','reference_token_capacity_status')},
        'input_tokens_observed': usage.get('input_tokens') if usage else None, 'output_tokens_observed': output_tokens,
        'reasoning_tokens_observed': reasoning_tokens, 'output_token_cap': cap,
        'output_cap_fraction_observed': output_tokens/cap if output_tokens is not None else None,
        'output_cap_remaining_tokens_observed': cap-output_tokens if output_tokens is not None else None,
        'max_operations': limits.max_operations, 'max_touched_voxels': limits.max_touched_voxels,
        'max_live_voxels': limits.max_voxels, 'max_json_bytes': limits.max_json_bytes,
        'reference_operation_fraction_of_local_limit': control['reference_operations']/limits.max_operations,
        'observed_operations': row.get('operation_count'), 'observed_touched_voxel_visits': None,
        'natural_stop': row.get('natural_stop'), 'completed_request_outcome': row['completed_request_outcome'],
        'usage_known': row['usage_known'], 'source_plan_identity': row['source_plan_identity'],
        'raw_response_sha256': row['artifacts'].get('raw.json',{}).get('sha256'),
        'completion_counterfactual': None}
    if row['status'] != 'success':
        require(row.get('metrics') is None and 'voxels.json' not in row['artifacts'], 'failed E must retain null metrics and no invented reconstruction')
        category = ('other' if not row['usage_known'] or not row['completed_request_outcome'] else
                    'truncation' if row.get('failure_kind')=='truncation' else 'validation' if row.get('failure_kind')=='validation_failure' else 'other')
        base['outcome_category'] = category
        capacity['outcome_category'] = category
        return base, capacity, []
    require(row['usage_known'] is True and row['completed_request_outcome'] is True, 'valid E requires identified completed known-usage outcome')
    shift = validate_alignment(row['metrics'])
    operations = validate_actions(read_json(package['artifacts']['parsed.json'])['operations'], limits)
    pred = payload_to_grid(read_json(package['artifacts']['voxels.json']), limits)
    executed = execute(operations, limits)
    require(executed.to_dict() == pred.to_dict(), 'saved E parsed actions do not reproduce saved voxels')
    require(len(operations)==row['operation_count'] and len(pred.voxels)==row['final_voxel_count'], 'saved operation/voxel count mismatch')
    worlds = {'absolute': pred.as_world().voxels,
              'aligned': {tuple(c[i]+shift[i] for i in range(3)):m for c,m in pred.as_world().voxels.items()}}
    independent, sensitivity, confusions = {}, {}, []
    for frame, world in worlds.items():
        metrics, pairs = independent_metrics(scene.grid.as_world().voxels, world)
        verify_metrics(metrics, row['metrics'][frame]); independent[frame] = metrics
        sensitivity[frame] = lexical_sensitivity(scene.grid.as_world().voxels, world)
        sensitivity[frame]['strict_material_aware_iou'] = metrics['material_aware_iou']
        sensitivity[frame]['lexical_material_iou_delta'] = sensitivity[frame]['material_aware_iou'] - metrics['material_aware_iou']
        for (gt_material,pred_material), count in sorted(pairs.items()):
            confusions.append({**{k:base[k] for k in ('scene_id','provider','global_schedule_index','attempt_id','input_id','input_sha256','gt_grid_sha256')},
                'frame':frame,'shift_xyz':[0,0,0] if frame=='absolute' else shift,
                'GT_material':gt_material,'prediction_material':pred_material,'voxel_count':count,
                'kind':'missing_GT_voxel' if pred_material=='air' else 'extra_prediction_voxel' if gt_material=='air'
                    else 'exact_match' if gt_material==pred_material else 'exact_ID_replacement'})
    transcription = operation_transcription(spec, operations)
    capacity['observed_touched_voxel_visits'] = transcription['touched_voxel_visits']
    aligned = independent['aligned']
    capacity['outcome_category'] = 'success'
    base.update(outcome_category='success', independent_metrics=independent, stored_metrics_match=True,
        stored_alignment_shift_xyz=shift, alignment_search_reexecuted=False, lexical_sensitivity=sensitivity,
        operation_transcription=transcription, parsed_to_saved_voxels_exact=True,
        near_perfect_occupancy_low_material=aligned['occupancy_iou']>=NEAR_PERFECT_OCCUPANCY and aligned['material_aware_iou']<LOW_EXACT_MATERIAL_IOU,
        same_material_set_with_spatial_replacements=aligned['material_set_f1']==1 and aligned['replace_count']>0)
    return base, capacity, confusions


def _write_json(path, value):
    with Path(path).open('xb') as stream:stream.write(json_bytes(value))


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n'); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _median(values):
    if not values:return None
    values=sorted(values); n=len(values)
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2


def capacity_summary(rows):
    result=[]
    for builder in BUILDERS:
        for category in ('success','truncation','validation','other'):
            selected=[r for r in rows if r['provider']==builder and r['outcome_category']==category]
            summary={'provider':builder,'outcome_category':category,'outcomes':len(selected),'independent_buildings':len({r['scene_id'] for r in selected})}
            for name in ('input_utf8_bytes','gt_occupied_voxels','cuboids','same_material_z_runs','reference_compact_json_bytes','input_tokens_observed','output_tokens_observed','reasoning_tokens_observed','output_cap_fraction_observed'):
                values=[r[name] for r in selected if r[name] is not None]
                summary[name]={'known':len(values),'minimum':min(values) if values else None,'median':_median(values),'maximum':max(values) if values else None}
            result.append(summary)
    return result


def run_exact_oracle(slots, goal007_manifest_path, destination, *, gate, specification_sha256):
    """Run after Phase0/specification freeze, under the caller's source guard."""
    started=time.monotonic(); check_gate(gate, specification_sha256)
    require(isinstance(specification_sha256,str) and re.fullmatch('[0-9a-f]{64}',specification_sha256), 'explicit expected specification SHA required')
    output=Path(destination).absolute()
    require(not output.exists(),'new E diagnostic directory required')
    if gate['input_kind']=='verified_sources':
        require(output.is_relative_to(Path('/root/minecraft-runs')), 'actual output must be a new research-run directory')
    packages, inventory=bind_sources(slots,goal007_manifest_path,gate=gate)
    _checked(ALIAS_SOURCE,ALIAS_SOURCE_SHA256,inventory)
    specification={'analysis_label':ANALYSIS_LABEL,'version':VERSION,'gate':gate,
        'fixed_near_perfect_occupancy_threshold':NEAR_PERFECT_OCCUPANCY,'fixed_low_material_iou_threshold':LOW_EXACT_MATERIAL_IOU,
        'lexical_aliases':LEXICAL_ALIASES,'lexical_source':str(ALIAS_SOURCE),'lexical_source_sha256':ALIAS_SOURCE_SHA256,
        'lexical_interpretation':'Narrow historical lexical grouping only, not verified physical equivalence. No fitting to GT or prediction, no slab/fence/plank/glass merges, no replacement of strict metrics.',
        'strict_metric_formula':'exact material match count / (GT count + prediction count - exact material match count)',
        'capacity_policy':'Observed provider token usage and original plan caps only; reference action count and UTF-8 size are not certified output-token requirements.',
        'controls_are_model_successes':False,'alignment_search_reexecuted':False,'research_api_requests_sent':0}
    output.mkdir(parents=True,exist_ok=False)
    _write_json(output/'E_specification_receipt.json',specification)
    outcomes, capacities, confusions, controls=[],[],[],[]
    by_scene=defaultdict(list)
    for package in packages:by_scene[package['slot']['scene_id']].append(package)
    for scene_id in sorted(by_scene):
        pair=by_scene[scene_id]; first=pair[0]
        require(all(p['gt']==first['gt'] and p['record']['input_text']==first['record']['input_text'] and p['limits']==first['limits'] for p in pair),
                'Builder E inputs/GT/limits differ within scene')
        scene=load_scene(first['gt']['building_directory'],Limits(**first['limits']))
        require(scene.scene_id==scene_id and object_hash(scene.grid.to_dict())==first['gt']['grid_sha256'],'loaded GT axis/origin/grid hash mismatch')
        spec,control=reference_control(scene,first['record']['input_text'],Limits(**first['limits']));controls.append(control)
        for package in pair:
            outcome,capacity,confusion=audit_outcome(package,scene,spec,control)
            outcomes.append(outcome);capacities.append(capacity);confusions.extend(confusion)
    after={path:sha256_file(regular(path)) for path in inventory}
    require(after==inventory,'immutable E source changed during diagnostic')
    summary={b:{category:sum(r['provider']==b and r['outcome_category']==category for r in outcomes)
                for category in ('success','truncation','validation','other')} for b in BUILDERS}
    focus=[r for r in outcomes if r['scene_id']=='buildings_100_v4_building_034']
    if gate['input_kind']=='verified_sources':
        require(len(controls)==60 and len(outcomes)==120 and summary=={'openai':{'success':55,'truncation':5,'validation':0,'other':0},'anthropic':{'success':46,'truncation':14,'validation':0,'other':0}},'frozen E outcome counts changed')
        require(len(focus)==2,'required building034 focus missing')
    taxonomy={'status':'E_AUDIT_PASS','analysis_label':ANALYSIS_LABEL,'outcomes':outcomes,'outcome_counts':summary,
        'fixed_near_perfect_low_material_cases':[{'scene_id':r['scene_id'],'provider':r['provider']} for r in outcomes if r['near_perfect_occupancy_low_material'] is True],
        'building034_focus':focus,'CPU_controls':len(controls),'controls_are_model_successes':False,
        'valid_strict_metric_checks':sum(r['outcome_category']=='success' for r in outcomes),
        'lexical_aliases':LEXICAL_ALIASES,'strict_metrics_changed':False,'capacity_group_summary':capacity_summary(capacities),
        'capacity_limit':'No claim that an unfinished output would become correct. Reference actions are one feasible partition, not a minimum, and their token requirement is unconfirmed.',
        'research_api_requests_sent':0,'api_key_queries':0,'token_count_api_calls':0,'local_tokenization_calls':0,'model_queries':0}
    _write_csv(output/'E_material_confusion.csv',confusions)
    _write_csv(output/'E_capacity_diagnostics.csv',capacities)
    _write_json(output/'E_reference_controls.json',controls)
    _write_json(output/'E_error_taxonomy.json',taxonomy)
    lines=['# Exact Oracleの材質・容量監査','',ANALYSIS_LABEL,'',
        f"E outcomes {len(outcomes)}件、GT→固定cuboid仕様→参照fill→CPU実行の完全一致control {len(controls)}建築。CPU controlをLLM成功へ加算しない。",'',
        'Builder別の成功・失敗: '+json.dumps(summary,ensure_ascii=False,sort_keys=True),'',
        '独立再計算は元の占有最適平行移動を固定して行い、absolute/alignedのstrict値と照合した。材質IoUは座標と厳密材質IDの組の集合IoUであり、材質集合F1とは異なる。元メトリクス、配置、原本応答を変更していない。','',
        'occupancy IoU >=0.99 かつ exact-material IoU <0.5 の固定診断条件に該当: '+str(len(taxonomy['fixed_near_perfect_low_material_cases']))+' outcomes。全例を保存し、この条件は母集団推定や因果的な閾値ではない。','']
    for row in focus:
        values=row['independent_metrics']['aligned'] if row['independent_metrics'] else None
        lines.append('building034 / '+row['provider']+': '+json.dumps({'status':row['outcome_category'],'aligned':values,'lexical_sensitivity':row['lexical_sensitivity']},ensure_ascii=False,sort_keys=True))
    lines+=['','転記診断はE paletteと参照cuboidの境界・出力operation材質を照合する。異なる正しい分割も可能なため、境界不一致だけを欠落とは呼ばない。最終voxel confusionを優先する。','',
        'alias感度は歴史的DIRECT_TYPE_NORMから計算前に固定した狭い語彙規則のみ。物理的同値の検証ではなく、stonebrickとbrick/brick_blockを同一視しない。slab/fence/plank/glassの広いまとめを使用しない。strict結果は置換せず別欄へ保存した。','',
        '容量欄は実際に保存されたprovider token usage、元planのoutput上限、入力byte、occupied数、cuboid数、同材質Z-run数、参照operation数・byteを記載する。byteをtokenへ換算せず、参照列がtoken上限へ入るかは未認証。理由tokenが判明する場合は別欄に残す。未完了出力の正しさを外挿しない。','',
        'Eは完全離散仕様から操作への診断であり自然言語の母集団上限ではない。成功だけの値は選択条件付きである。研究API・モデル・token-count・キー照会は0件。']
    with (output/'MATERIAL_VALIDITY_JA.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    products={p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    final={'status':'E_AUDIT_PASS','version':VERSION,'analysis_label':ANALYSIS_LABEL,'input_kind':gate['input_kind'],
        'gate':gate,'source_files':inventory,'source_files_before_after_unchanged':True,'code_sha256':sha256_file(__file__),
        'products':products,'outcomes':len(outcomes),'controls':len(controls),'outcome_counts':summary,
        'valid_strict_metric_checks':taxonomy['valid_strict_metric_checks'],'original_metrics_changed':False,
        'research_api_requests_sent':0,'api_key_queries':0,'token_count_api_calls':0,'local_tokenization_calls':0,
        'performance':{'seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    _write_json(output/'E_AUDIT_MANIFEST.json',final)
    return final
