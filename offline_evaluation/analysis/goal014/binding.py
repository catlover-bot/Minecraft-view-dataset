"""Read-only Goal014 joins to frozen scores and saved Goal010v2 surface F1@1.

No metric, model or statistical estimate is computed here. Normalized world-grid
hashes are inherited from the manifest-bound Goal010v2 receipt, while source
prediction/GT and stored alignment identities are independently cross-joined to
Goal007. These hash namespaces are deliberately not conflated.
"""
from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from ..goal013a import outcome_binding as old
from ..goal013a.associations import BUILDERS, finite, require

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'reports/server_only/goal013a'
GOAL013A_MANIFEST_SHA256 = 'ea08175659312e7cc667ca8e107f584de0c58bd0fc20ee062494d52ee845ca1e'
GOAL010V2_MANIFEST_SHA256 = 'f286aa20562b0954fc42ab479100b416066eedb0551ea4d9df34264773f98a14'
SURFACE_PRODUCT = '3_full/new_metric_values_960.json'
IDENTITIES = ('scene_id', 'description_model', 'provider', 'condition', 'global_schedule_index',
              'input_id', 'input_sha256', 'attempt_id', 'original_attempt_id', 'selection_reason',
              'source_label', 'source_plan_identity', 'source_entry_hash', 'source_gt_descriptor_hash',
              'gt_grid_sha256', 'status', 'failure_kind', 'usage_known', 'completed_request_outcome')


def object_hash(value, *, formatted=False):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      **({'indent': 2} if formatted else {'separators': (',', ':')}))
    return hashlib.sha256((text + ('\n' if formatted else '')).encode()).hexdigest()


def _sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'explicit SHA256 required')
    return value


def _index(rows, fields, label):
    result = {tuple(r[k] for k in fields): r for r in rows}
    require(len(result) == len(rows), label + ' duplicate identity')
    return result


def bind_rows(bound_rows, score_rows, slots, metric_rows, plans, *, fixture=False):
    """Pure join; fixture=True relaxes only fixed production population counts."""
    bound = _index(bound_rows, ('scene_id', 'description_provider'), 'bound scores')
    scores = _index(score_rows, ('scene_id', 'description_provider'), 'scores')
    require(set(bound) == set(scores), 'score/bound frame differs')
    sl = _index(slots, ('global_schedule_index',), 'slots')
    met = _index(metric_rows, ('global_schedule_index',), 'surface')
    require(set(sl) == set(met), 'surface/selected-slot frame differs')
    if not fixture:
        require(len(sl) == 960 and set(sl) == {(i,) for i in range(960)}, '960 selected slots required')
        require(len(bound) == 180, '180 planned score rows required')
    originals = {k: v for k, v in sl.items() if v['condition'] == 'O'}
    expected = {(s, p, b) for s, p in bound for b in BUILDERS}
    require({(s['scene_id'], s['description_model'], s['provider']) for s in originals.values()} == expected
            and len(originals) == len(expected), 'O source frame differs')
    results = []
    for key in sorted(bound):
        row, score = copy.deepcopy(bound[key]), scores[key]
        require(row['score'] == score['score'] and row['score_status'] == score['score_status'],
                'fixed score changed during outcome join')
        td = score['token_diagnostics']
        valid = score['score_status'] == 'COMPUTATION_VALID'
        if valid:
            require(td['token_coverage_status'] == 'VERIFIED' and td['phase2_token_eligible'] is True,
                    'score token coverage ineligible')
            require(type(td['truncated']) is bool, 'explicit boolean truncation flag required')
            require(td['summary_token_count_untruncated'] == row['summary_token_count_untruncated'],
                    'token length differs')
            require(td['truncated'] == (td['summary_dropped_tokens'] > 0), 'truncation/drop disagreement')
            row['truncated'] = td['truncated']
        else:
            require(score['score'] is None, 'missing score imputation forbidden')
            row['truncated'] = None
        for builder in BUILDERS:
            outcome = row['outcomes'][builder]
            idx = (outcome['global_schedule_index'],)
            require(idx in originals, 'O selected slot missing')
            slot, metric = originals[idx], met[idx]
            require((slot['scene_id'], slot['description_model'], slot['provider']) == (*key, builder),
                    'description/Builder identity mismatch')
            require(all(field in metric and metric[field] == slot.get(field) for field in IDENTITIES),
                    'surface/selected-slot identity differs')
            require(slot['input_id'] == row['O_input_id'] and slot['input_sha256'] == row['O_input_sha256'],
                    'O input binding differs')
            require(slot['attempt_id'] == slot['selected_attempt_id'] == outcome['selected_attempt_id']
                    and slot['selected_final_observation'] is True, 'selected attempt differs')
            require(metric['artifact_sha256'] == {k: v['sha256'] for k, v in slot['artifacts'].items()},
                    'prediction artifact hash differs')
            require(metric['original_metric_object_sha256'] == object_hash(slot['metrics']),
                    'original IoU object hash differs')
            plan = plans[slot['source_label']]
            require(plan['identity_hash'] == slot['source_plan_identity'] == object_hash(plan['identity'], formatted=True),
                    'source plan identity differs')
            gt = plan['identity']['gt_by_scene'][slot['scene_id']]
            require(object_hash(gt, formatted=True) == slot['source_gt_descriptor_hash'], 'GT descriptor differs')
            for field, gtfield in [('gt_grid_sha256', 'grid_sha256'), ('gt_voxels_sha256', 'voxels_sha256'),
                                   ('gt_bbox_sha256', 'bbox_sha256')]:
                require(metric[field] == gt[gtfield], 'GT source hash differs: ' + field)
            values = metric['metrics']['aligned']
            surface, iou = values['surface_f1_at_1'], outcome['aligned_occupancy_iou']
            require(surface is None or (finite(surface) and 0 <= surface <= 1), 'invalid surface scalar')
            null_reason = metric['null_reasons']['aligned'].get('surface_f1_at_1')
            require(surface is not None or (isinstance(null_reason, str) and null_reason), 'missing surface reason required')
            engine = metric['metric_provenance']
            if metric['metric_status'] == 'COMPUTED':
                require(slot['status'] == 'success' and values['occupancy_iou'] == iou,
                        'saved occupancy changed')
                shift = slot['metrics']['alignment']['shift_xyz']
                require(type(shift) is list and len(shift) == 3 and all(type(x) is int for x in shift),
                        'stored integer XYZ shift required')
                require(metric['shift_xyz'] == engine['stored_alignment_shift_xyz'] == shift,
                        'stored occupancy shift differs')
                require(engine['alignment_searched'] is False and engine['failure0_applied'] is False
                        and engine['protocol_id'] == 'goal010v2_fixed_shift_metrics_v1'
                        and engine['alignment_protocol'] == 'server_v2.2_strict_translation',
                        'fixed alignment protocol differs')
                require(engine['shift_sha256'] == object_hash(shift), 'stored shift hash differs')
                parts = [_sha(engine[k]) for k in ('gt_grid_sha256', 'pred_grid_sha256', 'shift_sha256')]
                require(engine['inputs_and_shift_sha256'] == hashlib.sha256(':'.join(parts).encode()).hexdigest(),
                        'normalized grid/shift hash binding differs')
                require(engine['implementation_sha256'] == metric['implementation_hashes'][
                    'offline_evaluation/analysis/goal010v2/metrics.py'], 'surface implementation hash differs')
            else:
                require(surface is None and engine is None, 'uncomputed surface must remain null')
            outcome.update(aligned_surface_f1_at_1=surface, surface_null_reason=null_reason,
                           surface_binding={'scalar_row_sha256': object_hash(metric),
                              'selected_attempt_id': slot['selected_attempt_id'],
                              'prediction_artifact_sha256': metric['artifact_sha256'].get('voxels.json'),
                              'gt_voxels_sha256': metric['gt_voxels_sha256'],
                              'gt_bbox_sha256': metric['gt_bbox_sha256'],
                              'source_gt_descriptor_hash': slot['source_gt_descriptor_hash'],
                              'original_metric_object_sha256': metric['original_metric_object_sha256'],
                              'metric_provenance': engine})
        results.append(row)
    if not fixture:
        counts = Counter(r['truncated'] for r in results)
        require(counts == {False: 119, True: 59, None: 2}, 'frozen truncation counts differ')
    return results


def load_inputs():
    """Resolve canonical products exclusively from fixed manifests and receipts."""
    manifest, md = old._read_bound(REPORT/'MANIFEST.json', GOAL013A_MANIFEST_SHA256)
    freeze = manifest['binding']['score_freeze']
    rows, scenes, previous = old.load_frozen_outcomes(freeze['path'], expected_freeze_sha256=freeze['sha256'])
    scores, sd = old._artifact(previous['score_artifact'])
    scores = old._rows(scores)
    source_manifest, _ = old._read_bound(old.GOAL007_MANIFEST, old.GOAL007_MANIFEST_SHA256)
    slots, slots_d = old._artifact(previous['selected_slots_artifact'])
    m10, m10d = old._read_bound(ROOT/'reports/server_only/goal010v2/MANIFEST.json', GOAL010V2_MANIFEST_SHA256)
    canonical = old.safe(Path(m10['canonical_run']))
    full, fd = old._read_bound(canonical/'3_full_RECEIPT.json', m10['canonical_receipts']['3_full_RECEIPT.json'])
    require(full['status'] == 'PASS' and full['source_before_after_unchanged'] is True
            and full['result']['stored_strict_shift_only'] is True and full['result']['no_new_alignment_search'] is True,
            'surface source/shift receipt failed')
    desc = full['products'][SURFACE_PRODUCT]
    metrics, metrics_d = old._read_bound(canonical/SURFACE_PRODUCT, desc['sha256'], desc['bytes'])
    for row in metrics:
        require(row['implementation_hashes'] == m10['analysis_binding']['hashes']
                and row['specification_sha256'] == m10['analysis_binding']['specification_sha256'],
                'surface fixed implementation/specification differs')
    plans, plan_descriptors = {}, {}
    for label in sorted({s['source_label'] for s in slots if s['condition'] == 'O'}):
        ref = source_manifest['sources'][label]
        plans[label], plan_descriptors[label] = old._read_bound(Path(ref['plan_path']), ref['plan_sha256'])
    token_expected = manifest['report_files']['TOKEN_COVERAGE_180.csv']
    token_raw = (REPORT/'TOKEN_COVERAGE_180.csv').read_bytes()
    require(len(token_raw) == token_expected['bytes'] and hashlib.sha256(token_raw).hexdigest() == token_expected['sha256'],
            'frozen token coverage CSV differs')
    token_rows = _index(list(csv.DictReader(io.StringIO(token_raw.decode()))), ('scene_id', 'description_provider'), 'token coverage')
    score_index = _index(scores, ('scene_id', 'description_provider'), 'scores')
    require(set(token_rows) == set(score_index), 'token CSV frame differs')
    for key, score in score_index.items():
        token = token_rows[key]; td = score['token_diagnostics']
        require(token['summary_sha256'] == score['summary_sha256'], 'token original summary hash differs')
        for field in ('truncated', 'summary_token_count_untruncated', 'token_ids_sha256'):
            expected = td.get(field)
            require(token[field] == ('' if expected is None else str(expected)), 'token CSV/score diagnostics differ')
    result = bind_rows(rows, scores, slots, metrics, plans)
    receipt = {'schema': 'goal014_outcome_binding_v1', 'status': 'PASS',
               'goal013a_manifest': md, 'goal010v2_manifest': m10d, 'surface_full_receipt': fd,
               'surface_artifact': metrics_d, 'score_artifact': sd, 'selected_slots_artifact': slots_d,
               'source_plans': plan_descriptors, 'goal013a_outcome_binding': previous,
               'planned_rows': len(result), 'planned_scenes': len(scenes), 'bound_O_outcomes': len(result)*2,
               'truncation_counts': {'false':119,'true':59,'missing':2},
               'surface_outcome': 'saved aligned surface_f1_at_1 at original occupancy-optimal shift',
               'GT_and_prediction_geometry_recomputed': False, 'world_grid_hashes':
               'Inherited exact recorded normalized hashes; source artifact/GT/shift identities cross-joined to Goal007',
               'source_writes': 0, 'new_statistics_computed': 0, 'network_requests': 0}
    return result, scenes, receipt
