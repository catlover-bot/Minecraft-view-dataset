"""Source-bound E diagnostic orchestration; only the caller writes products.

All original classifications, actions, voxels and scores stay immutable. The
caller first verifies the committed plan/code and holds the shared source locks.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import time

from ..goal010 import exact_oracle as old
from ..goal010v2.pipeline import SampleResourceLimit
from ...resmoke_v2 import adapter, schema
from ...server_v2.core import Limits, payload_to_grid, strict_json, validate_actions
from ...server_v2.data import load_scene
from . import material
from .tradeoff import _gate

VERSION = 'goal011_source_bound_material_batch_v1'
LABEL = 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC'
FRAMES = ('absolute', 'aligned')
KINDS = ('one_to_one', 'many_to_one')
CASE_CPU_SECONDS = 120
CASE_WALL_SECONDS = 180
PROGRESS_EVERY = 10
IDENTITY = ('scene_id', 'provider', 'global_schedule_index', 'attempt_id', 'input_id',
            'input_sha256', 'gt_grid_sha256', 'source_gt_descriptor_hash',
            'source_label', 'source_plan_identity', 'source_entry_hash')


class CaseResourceLimit(RuntimeError):
    pass


class MaterialBatchAuditFailure(ValueError):
    """Safe proof plus already completed diagnostic summaries; never raw text."""
    def __init__(self, error, state):
        resource_limit = isinstance(error, (SampleResourceLimit, CaseResourceLimit,
                                            material.MaterialResourceLimit, MemoryError))
        self.report = {'status': 'NOT_VERIFIED_RESOURCE_LIMIT' if resource_limit else 'FAIL',
                       'critical': not resource_limit,
                       'code': 'MATERIAL_RESOURCE_LIMIT_PARTIAL_PROOF' if resource_limit else 'MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED',
                       'exception_class': type(error).__name__,
                       'stage': state.get('stage'), 'current_identity': state.get('identity'),
                       'completed_outcome_summaries': copy.deepcopy(state.get('completed', [])),
                       'historical_changes_permitted': False,
                       'correction_proposal': ('Retain completed proof and report uncomputed cases; no numerical correction is established.'
                                               if resource_limit else
                                               'Inspect the bound source/control/count discrepancy; retain all historical files.')}
        # Validation exception strings may contain input fragments. The bound
        # identity and stage are sufficient to inspect the original privately.
        super().__init__('MATERIAL_SOURCE_CONTROL_OR_COUNT_AUDIT_FAILED')


@contextmanager
def case_budget():
    """Bound one complete control/case; retain and restore caller stage timers."""
    def stop(signum, frame):
        raise CaseResourceLimit('CASE_CPU_120_OR_WALL_180_SECONDS')
    handlers = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGPROF)}
    timers = {t: signal.getitimer(t) for t in (signal.ITIMER_REAL, signal.ITIMER_PROF)}
    started, cpu_started = time.monotonic(), time.process_time()
    for s in handlers:
        signal.signal(s, stop)
    try:
        for timer, maximum in ((signal.ITIMER_REAL, CASE_WALL_SECONDS),
                               (signal.ITIMER_PROF, CASE_CPU_SECONDS)):
            prior = timers[timer][0]
            signal.setitimer(timer, min(prior, maximum) if prior > 0 else maximum)
        yield
    finally:
        for timer in timers:
            signal.setitimer(timer, 0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
        for timer, (remaining, interval) in timers.items():
            if remaining > 0:
                elapsed = (time.monotonic() - started if timer == signal.ITIMER_REAL
                           else time.process_time() - cpu_started)
                signal.setitimer(timer, max(.001, remaining - elapsed), interval)


def _same(actual, expected, name):
    if actual is None or expected is None:
        good = actual is expected
    else:
        good = (type(actual) in (int, float) and type(expected) in (int, float)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12))
    old.require(good, 'new material diagnostic disagrees with independent original counts: ' + name)


def _original_raw_actions(item, parsed_actions, limits):
    """Read saved response text only; retain block spellings before normalization."""
    try:
        response = adapter.parse_response(item['slot']['provider'],
                                          Path(item['artifacts']['raw.json']).read_bytes())
    except adapter.ProviderError as exc:
        raise ValueError('saved successful raw E response cannot be parsed') from exc
    old.require(response.complete and response.failure_kind is None,
                'saved successful E raw response is incomplete')
    original_actions = schema.validate_document(strict_json(response.text, limits))
    old.require(validate_actions(original_actions, limits) == parsed_actions,
                'saved E raw action text differs from saved parsed actions')
    return original_actions


def _independent_confusion_check(result, independent_rows):
    """Compare every label pair to the frozen independent coordinate traversal.

    Aggregate C/M checks cannot detect a misplaced off-diagonal count. Convert
    the historical air sentinel only at this comparison boundary; the new
    confusion continues to distinguish absent geometry with a null side.
    """
    expected = Counter()
    for row in independent_rows:
        gt = None if row['GT_material'] == 'air' else row['GT_material']
        pred = None if row['prediction_material'] == 'air' else row['prediction_material']
        expected[gt, pred] += row['voxel_count']
    conf = result['confusion']
    gt_labels, pred_labels = conf['gt_labels'], conf['pred_labels']
    old.require(gt_labels == sorted({gt for gt, _ in expected if gt is not None})
                and pred_labels == sorted({pred for _, pred in expected if pred is not None}),
                'new material confusion vocabulary differs from independent source')
    observed = Counter()
    for row in conf['common_nonzero']:
        key = row['gt_material_id'], row['pred_material_id']
        old.require(key not in observed and key[0] in gt_labels and key[1] in pred_labels
                    and type(row['count']) is int and row['count'] > 0,
                    'invalid sparse common material confusion entry')
        observed[key] = row['count']
    counts = conf['common_counts']
    if counts is not None:
        old.require(len(counts) == len(gt_labels) and all(len(row) == len(pred_labels) for row in counts),
                    'incomplete Cartesian material confusion matrix')
        for i, gt in enumerate(gt_labels):
            for j, pred in enumerate(pred_labels):
                count = counts[i][j]
                old.require(type(count) is int and count >= 0 and count == observed[gt, pred],
                            'dense and sparse material confusion disagree')
    for group, category, labels, absent_side in (
            ('missing_by_gt_label', 'missing_geometry', gt_labels, 'pred_material_id'),
            ('extra_by_pred_label', 'extra_geometry', pred_labels, 'gt_material_id')):
        rows = conf[group]
        present_side = 'gt_material_id' if absent_side == 'pred_material_id' else 'pred_material_id'
        old.require([row[present_side] for row in rows] == labels, 'incomplete geometry confusion labels')
        for row in rows:
            old.require(row[absent_side] is None and row['category'] == category
                        and type(row['count']) is int and row['count'] >= 0,
                        'invalid missing/extra geometry confusion entry')
            observed[row['gt_material_id'], row['pred_material_id']] = row['count']
    old.require(observed == expected, 'new material confusion differs from independent per-coordinate counts')


def _independent_check(result, independent, independent_confusion):
    for key, original in (('g', 'gt_voxel_count'), ('p', 'pred_voxel_count'),
                          ('C', 'occupancy_intersection'), ('M', 'material_match_count'),
                          ('missing_occupied_count', 'add_count'),
                          ('extra_occupied_count', 'delete_count'),
                          ('common_material_mismatches', 'replace_count')):
        old.require(type(result[key]) is int and result[key] == independent[original],
                    'new material diagnostic integer count mismatch: ' + key)
    g, c, m = result['g'], result['C'], result['M']
    for key, value in [('occupancy_iou', independent['occupancy_iou']),
                       ('material_aware_iou', independent['material_aware_iou']),
                       ('conditional_exact_material_accuracy', m/c if c else None),
                       ('gt_material_recall', m/g if g else None)]:
        _same(result['strict'][key], value, key)
    _independent_confusion_check(result, independent_confusion)
    conf = result['confusion']
    shared = {(r['gt_material_id'], r['pred_material_id']): r['count'] for r in conf['common_nonzero']}
    for kind in KINDS:
        bound = result[kind]
        maximum = bound.get('maximum_exact_matches')
        if maximum is not None:
            old.require(type(maximum) is int and m <= maximum <= c, 'invalid mapping count bound')
            mapping = bound.get('mapping')
            if mapping is not None:
                old.require([r['pred_label'] for r in mapping] == conf['pred_labels'],
                            'mapping witness does not cover each predicted ID exactly once')
                targets = [r['gt_label'] for r in mapping if r['gt_label'] is not None]
                old.require(all(gt in conf['gt_labels'] for gt in targets)
                            and (kind != 'one_to_one' or len(targets) == len(set(targets))),
                            'mapping witness violates its target constraints')
                for row in mapping:
                    attained = shared.get((row['gt_label'], row['pred_label']), 0)
                    old.require(type(row['matched_count']) is int and row['matched_count'] == attained,
                                'mapping witness pair count differs from confusion')
                    if kind == 'many_to_one':
                        old.require(attained == max((shared.get((gt, row['pred_label']), 0)
                                                      for gt in conf['gt_labels']), default=0),
                                    'many-to-one witness does not attain its column optimum')
                old.require(sum(r['matched_count'] for r in mapping) == maximum,
                            'mapping witness does not attain its count')


def _confusion_rows(identity, frame, result):
    conf = result['confusion']
    counts = conf['common_counts']
    rows = []
    if counts is not None:
        for i, gt in enumerate(conf['gt_labels']):
            for j, pred in enumerate(conf['pred_labels']):
                rows.append({**identity, 'frame': frame, 'gt_material_id': gt,
                             'pred_material_id': pred, 'category': 'shared_occupancy',
                             'voxel_count': counts[i][j], 'complete_cartesian_matrix': True})
    else:
        rows.extend({**identity, 'frame': frame, **{k: r[k] for k in ('gt_material_id', 'pred_material_id')},
                     'category': 'shared_occupancy', 'voxel_count': r['count'],
                     'complete_cartesian_matrix': False} for r in conf['common_nonzero'])
    for group in ('missing_by_gt_label', 'extra_by_pred_label'):
        rows.extend({**identity, 'frame': frame,
                     **{k: r[k] for k in ('gt_material_id', 'pred_material_id', 'category')},
                     'voxel_count': r['count'], 'complete_cartesian_matrix': counts is not None}
                    for r in conf[group])
    return rows


def _public_frame(result):
    """Keep diagnostic counts and action hashes, never an action geometry copy."""
    value = copy.deepcopy(result)
    original_action = result['reference_action_audit']
    action = value['reference_action_audit'] = _public_action_audit(original_action)
    references, outputs = original_action.get('references', []), original_action.get('outputs', [])
    labels = Counter((r['reference_material'], r['predicted_material']) for r in references
                     if r['status'] == 'UNIQUE_LITERAL_BOUNDS_MATCH')
    action['unique_literal_material_pairs'] = [
        {'reference_material': a, 'predicted_material': b, 'operations': count,
         'label_agreement': a == b} for (a, b), count in sorted(labels.items())]
    action['ambiguity_reason_counts'] = dict(sorted(Counter(
        reason for r in references for reason in r['ambiguity_reasons']).items()))
    action['unclassified_predicted_operation_kinds'] = dict(sorted(Counter(
        r['operation'] for r in outputs if r['status'] == 'AMBIGUOUS_NOT_CLASSIFIED').items()))
    action['action_geometry_arrays_published'] = False
    action['per_action_geometry_remains_in_original_hashed_sources'] = True
    return value


def _flat_case(identity, frames, capacity, diagnostic_status='COMPUTED', reason=None):
    row = {**identity, 'original_status': 'success', 'diagnostic_status': diagnostic_status,
           'diagnostic_null_reason': reason, 'oracle_bound_is_model_success': False,
           'output_token_cap': capacity.get('output_token_cap'),
           'output_tokens_observed': capacity.get('output_tokens_observed'),
           'output_cap_fraction_observed': capacity.get('output_cap_fraction_observed'),
           'reference_operations': capacity.get('reference_operations'),
           'observed_operations': capacity.get('observed_operations'),
           'observed_touched_voxel_visits': capacity.get('observed_touched_voxel_visits')}
    for frame in FRAMES:
        result = frames.get(frame)
        for name in ('g', 'p', 'C', 'M', 'common_material_mismatches', 'missing_occupied_count', 'extra_occupied_count'):
            row[frame + '_' + name] = result.get(name) if result else None
        for name in ('occupancy_iou', 'conditional_exact_material_accuracy', 'gt_material_recall', 'material_aware_iou'):
            row[frame + '_' + name] = result['strict'].get(name) if result else None
        for kind in KINDS:
            bound = result[kind] if result else {}
            for name in ('status', 'maximum_exact_matches', 'conditional_accuracy_upper_bound',
                         'gt_material_recall_upper_bound', 'material_iou_upper_bound',
                         'additional_common_matches_over_original', 'residual_common_material_mismatches',
                         'recoverable_fraction_of_original_common_mismatches', 'ambiguous_mapping'):
                row[frame + '_' + kind + '_' + name] = bound.get(name)
    action = frames.get('absolute', {}).get('reference_action_audit', {})
    for name in ('status', 'unique_literal_pairs', 'unique_correct_label_pairs', 'unique_wrong_label_pairs',
                 'ambiguous_reference_actions', 'unclassified_predicted_actions'):
        row['action_' + name] = action.get(name)
    return row


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def summarize(outcomes):
    """Each E success contributes once per Builder; pooled voxels are separate."""
    rows = []
    for builder in old.BUILDERS:
        all_outcomes = [r for r in outcomes if r['provider'] == builder]
        successful = [r for r in all_outcomes if r['original_status'] == 'success']
        for frame in FRAMES:
            measured = [r['frames'][frame] for r in successful if frame in r['frames']]
            totals = {k: sum(r[k] for r in measured) for k in ('g', 'p', 'C', 'M')}
            result = {'provider': builder, 'frame': frame, 'planned_outcomes': len(all_outcomes),
                      'original_successes': len(successful),
                      'original_truncations': sum(r['failure_kind'] == 'truncation' for r in all_outcomes),
                      'measured_successes': len(measured),
                      'uncomputed_successes': len(successful) - len(measured),
                      'independent_buildings': len({r['scene_id'] for r in successful if frame in r['frames']}),
                      'pooled_voxel_counts': totals,
                      'pooled_M_over_C': totals['M']/totals['C'] if totals['C'] else None,
                      'pooled_M_over_g': totals['M']/totals['g'] if totals['g'] else None,
                      'conditional_success_only': True,
                      'equal_building_weight_within_builder': True,
                      'maps_fitted_separately_per_prediction_using_GT': True,
                      'perfect_occupancy_with_material_mismatch': sum(r['g'] == r['p'] == r['C'] and r['M'] < r['C'] for r in measured),
                      'near_perfect_occupancy_low_material': sum(r['strict']['occupancy_iou'] >= .99 and r['strict']['material_aware_iou'] < .5 for r in measured)}
            for key in ('occupancy_iou', 'conditional_exact_material_accuracy', 'gt_material_recall', 'material_aware_iou'):
                values = [r['strict'][key] for r in measured if r['strict'][key] is not None]
                result[key] = {'mean': _mean(values), 'valid_buildings': len(values),
                               'null_successes': len(measured) - len(values)}
            for kind in KINDS:
                bounds = [r[kind] for r in measured]
                result[kind] = {'status_counts': dict(sorted(Counter(r['status'] for r in bounds).items())),
                                'ambiguous_mappings': sum(r.get('ambiguous_mapping') is True for r in bounds),
                                'ambiguous_observed_label_mappings': sum(bool(r.get('ambiguous_observed_pred_labels')) for r in bounds)}
                for key in ('maximum_exact_matches', 'conditional_accuracy_upper_bound', 'gt_material_recall_upper_bound',
                            'material_iou_upper_bound', 'additional_common_matches_over_original',
                            'residual_common_material_mismatches', 'recoverable_fraction_of_original_common_mismatches'):
                    values = [r[key] for r in bounds if r.get(key) is not None]
                    result[kind][key] = {'mean': _mean(values), 'valid_buildings': len(values),
                                        'null_successes': len(measured) - len(values)}
            rows.append(result)
    return rows


def _public_action_audit(value):
    """Publish label/count/hash evidence without copying source cuboid geometry."""
    result = {k: copy.deepcopy(v) for k, v in value.items() if k not in ('references', 'outputs')}
    if 'references' not in value:
        return result
    pair_keys = ('reference_material', 'reference_material_as_supplied',
                 'predicted_material', 'predicted_material_as_supplied', 'label_agreement')
    pairs = Counter(tuple(row[key] for key in pair_keys) for row in value['references']
                    if row['status'] == 'UNIQUE_LITERAL_BOUNDS_MATCH')
    result['unique_literal_label_pair_counts'] = [dict(zip(pair_keys, key), action_pairs=count)
        for key, count in sorted(pairs.items(), key=lambda item: tuple(str(x) for x in item[0]))]
    labels = Counter((row['material_as_supplied'], row['material'], row['operation']) for row in value['outputs'])
    result['all_output_label_counts'] = [dict(zip(('material_as_supplied', 'canonical_material', 'operation'), key),
                                              action_count=count)
        for key, count in sorted(labels.items(), key=lambda item: tuple(str(x) for x in item[0]))]
    result['unclassified_reference_reason_counts'] = dict(sorted(Counter(
        reason for row in value['references'] for reason in row['ambiguity_reasons']).items()))
    result['action_coordinates_omitted_from_public_report'] = True
    result['literal_action_detail_status'] = 'VERIFIED_SOURCE_BOUND_DETAILS_NOT_PUBLISHED'
    return result


def render_report(result):
    audit = result['audit']
    lines = ['# E条件：固定geometry上の材質記号対応の診断', '', LABEL, '',
             f"元のE出力{audit['outcomes']}件を全件保持し、成功{audit['original_successes']}件と打切り失敗{audit['original_truncations']}件を分けた。"
             f"成功のうち新診断を計算できたのは{audit['diagnostic_computed_successes']}件。"
             f"GT→固定cuboid→参照操作のCPU完全一致controlは{audit['controls']}建築で、モデル成功へ加算しない。", '',
             '既存absoluteと保存済みoccupancy最適shiftのalignedを使用した。新しいalignmentやalias置換は行わず、元操作・voxel・評価値を変更していない。', '',
             '| Builder | frame | 成功 / 診断N | 元M/C | 単射上界M/C | 多対一上界M/C | 元M/g | 単射上界M/g |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    def fmt(value):
        return 'NULL' if value is None else f'{value:.6f}'
    for row in result['summary']:
        one, many = row['one_to_one'], row['many_to_one']
        lines.append(f"| {row['provider']} | {row['frame']} | {row['original_successes']} / {row['measured_successes']} | "
                     + ' | '.join(fmt(v) for v in [row['conditional_exact_material_accuracy']['mean'],
                         one['conditional_accuracy_upper_bound']['mean'], many['conditional_accuracy_upper_bound']['mean'],
                         row['gt_material_recall']['mean'], one['gt_material_recall_upper_bound']['mean']]) + ' |')
    lines += ['', '固定focusの保存済み原本に対する独立countsと上界:']
    for row in audit['building034_focus']:
        value = row['frames'].get('aligned')
        if value is not None:
            lines.append(f"- building_034 / {row['provider']}: g={value['g']}, p={value['p']}, C={value['C']}, M={value['M']}; "
                f"元M/C={fmt(value['strict']['conditional_exact_material_accuracy'])}, 元M/g={fmt(value['strict']['gt_material_recall'])}, "
                f"単射上界M={value['one_to_one'].get('maximum_exact_matches')}, 多対一上界M={value['many_to_one'].get('maximum_exact_matches')}。")
    lines += ['', '表は各Builderの成功建築を等重みで平均した値である。M/Cは共有占有内の厳密材質一致、M/gはGT全占有を分母とする。'
              'C=0ではM/CはNULLであり0補完しない。voxelを全件合算したΣM/ΣC・ΣM/Σgは別欄に保存し、この等重み平均と混同しない。', '',
              '対応上界は各予測ごとにGTを見て最大化した診断値である。単射は予測ラベルをGTラベルへ部分一対一対応させ、多対一はこの制約を緩める。'
              'geometryの欠落・余剰は変えられない。上界への改善は記号対応だけで説明し得る量であり、モデル内部の転記機構や原因の証明、修正済み出力、'
              '新しい成功例、学習した一般化対応ではない。最適対応の辞書順witnessと全最適解で可能な各ラベルの相手を保存し、同率・未観測ラベルを明示した。', '',
              '参照cuboidとの操作照合は絶対world XYZのinclusive boundsを使用する。参照・出力の両方で一意かつ非重複の場合だけ材質一致/不一致を分類する。'
              '重複・上書き・分割・併合・ずれ・carve・対応不明は曖昧のまま残す。別分割が同じ最終geometryを作れるため、境界不一致を欠落や意図的shiftとは断定しない。', '',
              '保存raw responseのaction本文を凍結parser/schemaで再検証し、parsed操作と一致することを確認した。材質IDは凍結executorの規約に従う'
              '（任意minecraft:接頭辞を除去、air類は欠損）。rawのblock綴りとcanonical IDを分けて保存し、stonebrick/stone_bricks等の同義語補正はしない。'
              '公開JSONでは操作座標を複製せず、一意対応のlabel pair件数・全操作のraw/canonical ID頻度・source hashを保存する。', '',
              'building_034/OpenAI/Eは結果によらず固定focusとして監査JSONへ収録した。全101成功の表・全confusion・両上界を保持し、良い対応が得られた例だけを選択しない。', '',
              '出力上限は元plan、token数は保存済みprovider usageのみ。参照operation数・UTF-8 byte数をtokenへ換算せず、未完了出力の正しさや上限内完了を推測しない。'
              '19件の打切りはgeometry/material診断の分母から除き、失敗として別記録に保持した。', '',
              '成功例条件付きの診断であり、両Builderの成功集合差を無視した因果比較は行わない。研究API・モデル・token-count・キー照会は0件。', '']
    return '\n'.join(lines)


def run_material(package, gate):
    """Return CSV-ready tables and audit JSON under a frozen verified gate.

    package is the verified source bundle augmented with goal007_manifest_path
    and old_e_controls_descriptor={path,sha256,bytes}. It is never mutated.
    The caller binds these descriptors to immutable historical manifests and
    holds source locks. This function independently hashes every source it reads.
    """
    state = {'stage': 'gate', 'identity': None, 'completed': []}
    try:
        return _run_material(package, gate, state)
    except (ValueError, KeyError, SampleResourceLimit, CaseResourceLimit, material.MaterialResourceLimit, MemoryError) as exc:
        if hasattr(exc, 'report'):
            raise
        raise MaterialBatchAuditFailure(exc, state) from exc


def _run_material(package, gate, state):
    _gate(gate, gate.get('plan_sha256') if isinstance(gate, dict) else None)
    old.check_gate(gate, gate['plan_sha256'])
    state['stage'] = 'source_binding'
    started, cpu_started = time.monotonic(), time.process_time()
    before = old.object_hash(package)
    packages, inventory = old.bind_sources(package['slots'], package['goal007_manifest_path'], gate=gate)
    manifest = old.read_json(package['goal007_manifest_path'])
    old.require(package['goal007_manifest'] == manifest, 'in-memory Goal007 manifest differs from bound file')
    descriptor = package['old_e_controls_descriptor']
    control_path = old._checked(descriptor['path'], descriptor['sha256'], inventory, descriptor['bytes'])
    old_controls = old.read_json(control_path)
    controls_by_scene = {r['scene_id']: r for r in old_controls}
    old.require(len(controls_by_scene) == len(old_controls), 'duplicate historical E control')
    by_scene = defaultdict(list)
    for item in packages:
        by_scene[item['slot']['scene_id']].append(item)
    old.require(set(controls_by_scene) == set(by_scene), 'historical E control scope differs')
    outcomes, controls, cases, confusion, mappings, capacities = [], [], [], [], [], []
    for scene_id in sorted(by_scene):
        state.update(stage='reference_control', identity={'scene_id': scene_id})
        pair = by_scene[scene_id]
        first = pair[0]
        old.require(all(p['gt'] == first['gt'] and p['record']['input_text'] == first['record']['input_text']
                        and p['limits'] == first['limits'] for p in pair), 'E inputs/GT/limits differ between Builders')
        limits = Limits(**first['limits'])
        with case_budget():
            scene = load_scene(first['gt']['building_directory'], limits)
            old.require(scene.scene_id == scene_id and old.object_hash(scene.grid.to_dict()) == first['gt']['grid_sha256'],
                        'loaded E GT origin/grid differs from source binding')
            spec, control = old.reference_control(scene, first['record']['input_text'], limits)
            old.require(control == controls_by_scene[scene_id], 'historical E reference control changed')
            reference = old.cuboids.reference_actions(spec, limits)['operations']
        controls.append(control)
        for item in pair:
            row = item['slot']
            identity = {k: row[k] for k in IDENTITY}
            state.update(stage='case', identity=identity)
            record = {**identity, 'original_status': row['status'], 'failure_kind': row.get('failure_kind'),
                      'frames': {}, 'model_success_reclassified': False}
            outcome, capacity = None, {}
            try:
                with case_budget():
                    outcome, capacity, independent_confusion = old.audit_outcome(item, scene, spec, control)
                    if row['status'] == 'success':
                        actions = validate_actions(old.read_json(item['artifacts']['parsed.json'])['operations'], limits)
                        original_actions = _original_raw_actions(item, actions, limits)
                        pred = payload_to_grid(old.read_json(item['artifacts']['voxels.json']), limits).as_world().voxels
                        shift = old.validate_alignment(row['metrics'])
                        binding = {**identity, 'artifact_sha256': outcome['artifact_sha256'],
                                   'reference_actions_sha256': control['reference_actions_sha256'],
                                   'original_strict_metrics_sha256': old.object_hash(row['metrics']),
                                   'controls_bound_to_previous_CPU_result': True}
                        absolute = material.analyze_case(scene.grid.as_world().voxels, pred, [0, 0, 0],
                            reference_actions=reference, predicted_actions=original_actions, source_binding=binding, limits=limits)
                        aligned = absolute if shift == [0, 0, 0] else material.analyze_case(
                            scene.grid.as_world().voxels, pred, shift, reference_actions=reference,
                            predicted_actions=original_actions, source_binding=binding, limits=limits)
                        for frame, value in (('absolute', absolute), ('aligned', aligned)):
                            _independent_check(value, outcome['independent_metrics'][frame],
                                               [r for r in independent_confusion if r['frame'] == frame])
                            record['frames'][frame] = _public_frame(value)
                        complete = all(value[kind].get('maximum_exact_matches') is not None
                                       for value in record['frames'].values() for kind in KINDS)
                        complete = complete and all(value['reference_action_audit']['status'] == 'COMPUTED_LITERAL_BOUNDARY_AUDIT'
                                                    for value in record['frames'].values())
                        record.update(diagnostic_status='COMPUTED' if complete else 'COMPUTED_WITH_OPTIONAL_MISSING',
                                      original_metric_counts_independently_verified=True,
                                      complete_confusion_independently_verified=True,
                                      raw_actions_to_saved_parsed_exact=True,
                                      parsed_to_saved_voxels_exact=True, stored_alignment_shift_xyz=shift)
                    else:
                        record.update(diagnostic_status='NOT_EVALUATED_ORIGINAL_FAILURE',
                                      original_metric_counts_independently_verified=None,
                                      complete_confusion_independently_verified=None,
                                      raw_actions_to_saved_parsed_exact=None,
                                      parsed_to_saved_voxels_exact=None)
            except (CaseResourceLimit, material.MaterialResourceLimit, MemoryError) as exc:
                record.update(frames={}, diagnostic_status='NOT_COMPUTED_RESOURCE_LIMIT',
                              diagnostic_null_reason=getattr(exc, 'reason', type(exc).__name__),
                              original_metric_counts_independently_verified=outcome is not None,
                              complete_confusion_independently_verified=None,
                              raw_actions_to_saved_parsed_exact=None,
                              parsed_to_saved_voxels_exact=outcome is not None and row['status'] == 'success')
            record['capacity'] = capacity
            capacities.append(capacity)
            if row['status'] == 'success':
                cases.append(_flat_case(identity, record['frames'], capacity, record['diagnostic_status'], record.get('diagnostic_null_reason')))
                for frame in FRAMES:
                    value = record['frames'].get(frame)
                    if value is not None:
                        confusion.extend(_confusion_rows(identity, frame, value))
                    for kind in KINDS:
                        mappings.append({**identity, 'frame': frame, 'mapping_kind': kind,
                            **(value[kind] if value else {'status': 'NOT_COMPUTED_RESOURCE_LIMIT',
                                'reason': record.get('diagnostic_null_reason'), 'maximum_exact_matches': None, 'mapping': None}),
                            'geometry_modified': False, 'model_success_reclassified': False})
            outcomes.append(record)
            state['completed'].append({**identity, 'original_status': record['original_status'],
                'diagnostic_status': record['diagnostic_status'],
                'frame_counts': {frame: {k: value[k] for k in ('g', 'p', 'C', 'M')}
                                 for frame, value in record['frames'].items()}})
            if len(outcomes) % PROGRESS_EVERY == 0 or len(outcomes) == len(packages):
                print(json.dumps({'event': 'goal011_material_progress', 'outcomes_completed': len(outcomes),
                                  'outcomes_total': len(packages), 'success_cases_retained': len(cases)},
                                 sort_keys=True), flush=True)
    state.update(stage='final_source_and_scope_audit', identity=None)
    counts = {builder: dict(sorted(Counter(
        'success' if r['original_status'] == 'success' else r['failure_kind'] or 'other'
        for r in outcomes if r['provider'] == builder).items())) for builder in old.BUILDERS}
    if gate['input_kind'] == 'verified_sources':
        old.require(len(controls) == 60 and len(outcomes) == 120 and len(cases) == 101
                    and counts == {'openai': {'success': 55, 'truncation': 5},
                                   'anthropic': {'success': 46, 'truncation': 14}}, 'fixed E 101/19 outcome scope changed')
    global_counts = Counter()
    for row in confusion:
        global_counts[tuple(row[k] for k in ('provider', 'frame', 'category', 'gt_material_id', 'pred_material_id'))] += row['voxel_count']
    global_confusion = [dict(zip(('provider', 'frame', 'category', 'gt_material_id', 'pred_material_id'), key),
                             voxel_count=value, weighting='POOLED_VOXELS_NOT_EQUAL_BUILDING')
                        for key, value in sorted(global_counts.items(), key=lambda item: tuple('' if v is None else v for v in item[0]))]
    focus = [r for r in outcomes if r['scene_id'] == 'buildings_100_v4_building_034']
    if gate['input_kind'] == 'verified_sources':
        old.require(len(focus) == 2 and {r['provider'] for r in focus} == set(old.BUILDERS), 'required building034 proof missing')
    after = {path: old.sha256_file(old.regular(path)) for path in inventory}
    old.require(after == inventory, 'immutable E source changed during material audit')
    old.require(old.object_hash(package) == before, 'input package mutated during material audit')
    missing = sum(r['diagnostic_status'] != 'COMPUTED' for r in outcomes if r['original_status'] == 'success')
    audit = {'status': 'PASS' if missing == 0 else 'PASS_WITH_DIAGNOSTIC_RESOURCE_MISSING',
             'outcomes': len(outcomes), 'controls': len(controls), 'controls_are_model_successes': False,
             'original_successes': len(cases), 'original_truncations': sum(r['failure_kind'] == 'truncation' for r in outcomes),
             'outcome_counts': counts, 'diagnostic_computed_successes': len(cases)-missing,
             'diagnostic_resource_missing_successes': missing,
             'diagnostic_case_count_unit': 'original_successful_prediction_not_frame',
             'optional_missing_successes': sum(r['diagnostic_status'] == 'COMPUTED_WITH_OPTIONAL_MISSING' for r in outcomes),
             'outcome_records': outcomes, 'controls_records': controls,
             'capacity_group_summary': old.capacity_summary([r for r in capacities if r]),
             'building034_focus': focus, 'source_before_after_unchanged': True,
             'historical_control_equality_verified': True,
             'successes_and_failures_never_reclassified': True,
             'original_metrics_changed': False, 'alignment_search_reexecuted': False,
             'new_confidence_intervals': 0, 'new_p_values': 0, 'research_api_requests_sent': 0,
             'model_queries': 0, 'token_count_api_requests': 0, 'api_key_queries': 0,
             'performance': {'wall_seconds': time.monotonic()-started, 'cpu_seconds': time.process_time()-cpu_started}}
    result = {'version': VERSION, 'analysis_label': LABEL, 'cases': cases, 'confusion': confusion,
              'global_confusion': global_confusion, 'mapping': mappings, 'audit': audit,
              'summary': summarize(outcomes), 'capacity': capacities, 'source_inventory': inventory,
              'provenance': {'gate': copy.deepcopy(gate), 'input_bundle_sha256': before,
                             'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'material_implementation_sha256': material.implementation_sha256(),
                             'caller_verifies_committed_code_and_holds_source_locks': True}}
    audit['summary'] = result['summary']
    audit['provenance'] = result['provenance']
    audit['pooled_global_confusion'] = global_confusion
    result['report_markdown'] = render_report(result)
    return result
