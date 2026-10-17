"""GT-informed label-binding upper bounds on immutable, fixed E geometry.

Pure CPU diagnostics: no source loading, alignment search, artifact correction,
API, or file writes. The caller verifies source controls and frozen plan/code.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

from ...server_v2.core import Limits, material_id, validate_actions

PROTOCOL_ID = 'goal011_fixed_geometry_material_binding_v1'
MAX_LABELS = 64
MAX_ASSIGNMENT_CALLS = 8192
CPU_BUDGET_SECONDS = 120.0


class MaterialResourceLimit(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class _Budget:
    def __init__(self):
        self.start = time.process_time()

    def check(self):
        if time.process_time() - self.start > CPU_BUDGET_SECONDS:
            raise MaterialResourceLimit('CPU_BUDGET_EXCEEDED')


def implementation_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _validate_world(grid, limits, budget):
    if type(grid) is not dict:
        raise ValueError('canonical world dictionary required')
    if len(grid) > limits.max_voxels:
        raise MaterialResourceLimit('OCCUPIED_LIMIT')
    for index, (point, label) in enumerate(grid.items()):
        if (type(point) is not tuple or len(point) != 3 or any(type(x) is not int for x in point)
                or any(abs(x) > limits.max_coordinate_abs for x in point)):
            raise ValueError('bounded integer world XYZ tuple required')
        if not isinstance(label, str) or material_id(label) != label or label == 'air':
            raise ValueError('canonical exact non-air material ID required')
        if index % 8192 == 0:
            budget.check()


def _grid_hash(grid):
    digest = hashlib.sha256(b'goal011_canonical_world_XYZ_v1\n')
    for c in sorted(grid):
        digest.update(json.dumps([*c, grid[c]], separators=(',', ':'), ensure_ascii=True).encode() + b'\n')
    return digest.hexdigest()


def _score_fields(count, g, p, c, original):
    return {'maximum_exact_matches': count,
            'conditional_accuracy_upper_bound': count / c if c else None,
            'gt_material_recall_upper_bound': count / g if g else None,
            'material_iou_upper_bound': count / (g + p - count) if g + p else 1.0,
            'additional_common_matches_over_original': count - original,
            'residual_common_material_mismatches': c - count,
            'recoverable_fraction_of_original_common_mismatches': (count - original) / (c - original) if c > original else None,
            'null_reasons': dict((key, reason) for key, reason, missing in [
                ('conditional_accuracy_upper_bound', 'NO_COMMON_OCCUPANCY', c == 0),
                ('gt_material_recall_upper_bound', 'EMPTY_GT', g == 0),
                ('recoverable_fraction_of_original_common_mismatches', 'NO_ORIGINAL_COMMON_MATERIAL_MISMATCHES', c == original),
            ] if missing)}


def mapping_upper_bounds(gt_labels, pred_labels, common_counts, *, g, p, original_matches, budget=None):
    """Return exact one-to-one and many-to-one oracle bounds, never predictions.

    common_counts is GT-major with all zero cells explicit. One-to-one maps
    sorted predicted IDs injectively to GT IDs or None (unmatched). Tied optimum
    witnesses use sorted GT IDs before None, lexicographically by predicted ID.
    Feasible target sets are global across *all* optimal mappings, not merely
    choices conditional on the selected prefix. Dummy-column permutations are
    ignored when determining ambiguity.
    """
    budget = budget or _Budget()
    if (not isinstance(gt_labels, list) or not isinstance(pred_labels, list)
            or any(not isinstance(x, str) for x in gt_labels + pred_labels)
            or gt_labels != sorted(set(gt_labels)) or pred_labels != sorted(set(pred_labels))):
        raise ValueError('unique sorted exact label vocabularies required')
    if len(gt_labels) > MAX_LABELS or len(pred_labels) > MAX_LABELS:
        raise MaterialResourceLimit('LABEL_LIMIT')
    if (type(common_counts) is not list or len(common_counts) != len(gt_labels)
            or any(type(row) is not list or len(row) != len(pred_labels)
                   or any(type(x) is not int or x < 0 for x in row) for row in common_counts)):
        raise ValueError('nonnegative integer complete confusion matrix required')
    c = sum(map(sum, common_counts))
    if (any(type(x) is not int or x < 0 for x in (g, p, original_matches))
            or not original_matches <= c <= min(g, p)):
        raise ValueError('confusion and geometry counts disagree')
    diagonal = sum(common_counts[i][j] for i, a in enumerate(gt_labels)
                   for j, b in enumerate(pred_labels) if a == b)
    if original_matches != diagonal:
        raise ValueError('original exact matches differ from strict confusion diagonal')
    if c == 0:
        undefined = {**_score_fields(0, g, p, c, original_matches),
                     'status': 'UNIDENTIFIABLE_NO_COMMON_OCCUPANCY', 'mapping': None,
                     'ambiguous_mapping': None, 'ambiguity_reason': 'NO_OBSERVED_LABEL_BINDING',
                     'optimal_mapping_count': None, 'optimal_mapping_count_status': 'NOT_ENUMERATED'}
        return {'one_to_one': dict(undefined), 'many_to_one': dict(undefined)}
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    weights = [[common_counts[i][j] for i in range(len(gt_labels))] for j in range(len(pred_labels))]
    calls = 0

    def optimum(row_ids, gt_ids):
        nonlocal calls
        budget.check()
        calls += 1
        if calls > MAX_ASSIGNMENT_CALLS:
            raise MaterialResourceLimit('ASSIGNMENT_CALL_LIMIT')
        if not row_ids:
            return 0
        # One dummy per remaining prediction allows unmatched rows. Their ID
        # permutations carry no meaning, have zero weight, and are not reported.
        matrix = np.zeros((len(row_ids), len(gt_ids) + len(row_ids)), dtype=np.int64)
        for i, row in enumerate(row_ids):
            for j, col in enumerate(gt_ids):
                matrix[i, j] = weights[row][col]
        ri, ci = linear_sum_assignment(matrix, maximize=True)
        return int(matrix[ri, ci].sum())

    all_rows, all_gt = list(range(len(pred_labels))), list(range(len(gt_labels)))
    best = optimum(all_rows, all_gt)
    selected, available, fixed = [], list(all_gt), 0
    for row in all_rows:
        for col in [*available, None]:
            increment = weights[row][col] if col is not None else 0
            rest_gt = [x for x in available if x != col]
            if fixed + increment + optimum(all_rows[row + 1:], rest_gt) == best:
                selected.append(col)
                available = rest_gt
                fixed += increment
                break
        else:
            raise RuntimeError('no exact lexicographic optimum witness')
    assert fixed == best
    possibilities = []
    for row in all_rows:
        possible = []
        rest_rows = [x for x in all_rows if x != row]
        for col in [*all_gt, None]:
            increment = weights[row][col] if col is not None else 0
            if increment + optimum(rest_rows, [x for x in all_gt if x != col]) == best:
                possible.append(col)
        possibilities.append(possible)
    one_mapping = [{'pred_label': pred_labels[row],
                    'gt_label': gt_labels[col] if col is not None else None,
                    'matched_count': weights[row][col] if col is not None else 0,
                    'observed_common_pred_count': sum(weights[row]),
                    'possible_gt_labels_in_any_optimum': [gt_labels[x] if x is not None else None for x in possibilities[row]],
                    'ambiguous': len(possibilities[row]) > 1}
                   for row, col in enumerate(selected)]
    many_mapping, many_score = [], 0
    for row in all_rows:
        maximum = max(weights[row], default=0)
        options = [i for i in all_gt if weights[row][i] == maximum]
        if maximum == 0:
            options.append(None)
        chosen = options[0]
        many_score += maximum
        many_mapping.append({'pred_label': pred_labels[row],
                             'gt_label': gt_labels[chosen] if chosen is not None else None,
                             'matched_count': maximum, 'observed_common_pred_count': sum(weights[row]),
                             'possible_gt_labels_in_any_optimum': [gt_labels[i] if i is not None else None for i in options],
                             'ambiguous': len(options) > 1})
    assert original_matches <= best <= many_score <= c
    result = {}
    for name, mapping, maximum in [('one_to_one', one_mapping, best), ('many_to_one', many_mapping, many_score)]:
        result[name] = {**_score_fields(maximum, g, p, c, original_matches),
                        'status': 'COMPUTED_GT_INFORMED_ORACLE_BOUND', 'mapping': mapping,
                        'ambiguous_mapping': any(x['ambiguous'] for x in mapping),
                        'ambiguous_pred_labels': [x['pred_label'] for x in mapping if x['ambiguous']],
                        'ambiguous_observed_pred_labels': [x['pred_label'] for x in mapping if x['ambiguous'] and x['observed_common_pred_count'] > 0],
                        'unobserved_pred_labels': [x['pred_label'] for x in mapping if x['observed_common_pred_count'] == 0],
                        'optimal_mapping_count': None, 'optimal_mapping_count_status': 'NOT_ENUMERATED',
                        'lexicographic_rule': 'predicted IDs sorted; target GT IDs sorted before unmatched None',
                        'geometry_unchanged': True, 'artifact_modified': False,
                        'interpretation': 'GT-informed diagnostic upper bound, not a corrected model output or alias map'}
    result['one_to_one']['assignment_solver_calls'] = calls
    budget.check()
    return result


def _bounds(action):
    if action['op'] == 'set':
        return tuple(action[a] for a in 'xyz') * 2
    return tuple(action[a + suffix] for suffix in ('1', '2') for a in 'xyz')


def _overlapping_actions(actions, budget):
    owners, overlapping = {}, set()
    visits = 0
    for index, action in enumerate(actions):
        bounds = _bounds(action)
        for point in itertools.product(*(range(bounds[i], bounds[i + 3] + 1) for i in range(3))):
            prior = owners.get(point)
            if prior is not None:
                overlapping.add(prior)
                overlapping.add(index)
            else:
                owners[point] = index
            visits += 1
            if visits % 8192 == 0:
                budget.check()
    budget.check()
    return overlapping


def audit_reference_actions(reference_actions, predicted_actions, *, limits=Limits(), budget=None):
    """Literal absolute-world matching only; no inferred split/merge/shift intent.

    A pair is classified only if both lists have exactly one identical inclusive
    bound and neither action footprint overlaps any other action in its list.
    The latter rule avoids attributing final voxel labels to overwritten writes.
    Actions are validated by the frozen validator, including endpoint ordering.
    """
    budget = budget or _Budget()
    refs = validate_actions(reference_actions, limits)
    pred = validate_actions(predicted_actions, limits)
    if any(a['op'] == 'carve' or a['block'] == 'air' for a in refs):
        raise ValueError('reference control must contain non-air writes only')
    ref_index, pred_index = defaultdict(list), defaultdict(list)
    for i, a in enumerate(refs):
        ref_index[_bounds(a)].append(i)
    for i, a in enumerate(pred):
        pred_index[_bounds(a)].append(i)
    ref_overlap = _overlapping_actions(refs, budget)
    pred_overlap = _overlapping_actions(pred, budget)
    rows, mapped_pred = [], set()
    for i, action in enumerate(refs):
        bounds = _bounds(action)
        candidates = pred_index.get(bounds, [])
        reasons = []
        if len(ref_index[bounds]) != 1:
            reasons.append('DUPLICATE_REFERENCE_BOUNDS')
        if len(candidates) == 0:
            reasons.append('NO_EXACT_LITERAL_BOUNDARY_MATCH')
        elif len(candidates) != 1:
            reasons.append('DUPLICATE_PREDICTED_BOUNDS')
        if i in ref_overlap:
            reasons.append('OVERLAPPING_REFERENCE_ACTIONS')
        if any(j in pred_overlap for j in candidates):
            reasons.append('OVERLAPPING_OR_OVERWRITTEN_PREDICTED_ACTIONS')
        if len(candidates) == 1 and (pred[candidates[0]]['op'] == 'carve' or pred[candidates[0]]['block'] == 'air'):
            reasons.append('DELETION_IS_NOT_MATERIAL_BINDING')
        matched = candidates[0] if not reasons else None
        if matched is not None:
            mapped_pred.add(matched)
        rows.append({'reference_action_index': i, 'bounds_xyz_inclusive': list(bounds),
                     'reference_material': action['block'], 'candidate_predicted_action_indices': candidates,
                     'reference_material_as_supplied': reference_actions[i].get('block'),
                     'predicted_action_index': matched,
                     'predicted_material': pred[matched]['block'] if matched is not None else None,
                     'predicted_material_as_supplied': predicted_actions[matched].get('block') if matched is not None else None,
                     'label_agreement': action['block'] == pred[matched]['block'] if matched is not None else None,
                     'status': 'UNIQUE_LITERAL_BOUNDS_MATCH' if matched is not None else 'AMBIGUOUS_NOT_CLASSIFIED',
                     'ambiguity_reasons': reasons})
    budget.check()
    unique = [x for x in rows if x['status'] == 'UNIQUE_LITERAL_BOUNDS_MATCH']
    output = [{'predicted_action_index': i, 'operation': a['op'],
               'bounds_xyz_inclusive': list(_bounds(a)), 'material': a.get('block'),
               'material_as_supplied': predicted_actions[i].get('block'),
               'status': 'UNIQUE_LITERAL_BOUNDS_MATCH' if i in mapped_pred else 'AMBIGUOUS_NOT_CLASSIFIED'}
              for i, a in enumerate(pred)]
    return {'status': 'COMPUTED_LITERAL_BOUNDARY_AUDIT', 'coordinate_system': 'absolute_world_XYZ',
            'alignment_applied_to_actions': False, 'reference_actions': len(refs), 'predicted_actions': len(pred),
            'unique_literal_pairs': len(unique), 'unique_correct_label_pairs': sum(x['label_agreement'] for x in unique),
            'unique_wrong_label_pairs': sum(not x['label_agreement'] for x in unique),
            'ambiguous_reference_actions': len(refs) - len(unique),
            'unclassified_predicted_actions': len(pred) - len(mapped_pred),
            'reference_overlap_actions': len(ref_overlap), 'predicted_overlap_actions': len(pred_overlap),
            'references': rows, 'outputs': output,
            'reference_actions_sha256': _hash(refs), 'predicted_actions_sha256': _hash(pred),
            'reference_actions_as_supplied_sha256': _hash(reference_actions),
            'predicted_actions_as_supplied_sha256': _hash(predicted_actions),
            'material_id_policy': 'Frozen executor canonical IDs; optional minecraft: prefix removed and air variants denote absence; exact supplied spellings retained separately; no synonym rewriting',
            'no_exact_match_does_not_prove_omission_or_shift': True,
            'alternative_partitions_may_execute_to_identical_geometry': True,
            'model_internal_mechanism_inferred': False}


def analyze_case(gt_world_dict, pred_world_dict, shift_xyz, *, reference_actions=None,
                 predicted_actions=None, source_binding=None, limits=Limits()):
    """Audit one verified successful E output; retain non-successes outside here.

    Caller binds all 101 valid E outcomes and keeps 19 truncations separately.
    Reference decoding/round-trip and original raw/parsed/voxel consistency must
    already pass frozen source checks. No truncation is reconstructed here.
    """
    budget = _Budget()
    _validate_world(gt_world_dict, limits, budget)
    _validate_world(pred_world_dict, limits, budget)
    if (not isinstance(shift_xyz, (list, tuple)) or len(shift_xyz) != 3
            or any(type(x) is not int or abs(x) > 2 * limits.max_coordinate_abs for x in shift_xyz)):
        raise ValueError('saved strict XYZ integer shift required')
    if source_binding is not None and not isinstance(source_binding, dict):
        raise ValueError('source binding metadata must be a dictionary')
    shift = tuple(shift_xyz)
    gt = gt_world_dict
    pred = {tuple(point[i] + shift[i] for i in range(3)): value for point, value in pred_world_dict.items()}
    gl, pl = sorted(set(gt.values())), sorted(set(pred.values()))
    common, missing, extra = Counter(), Counter(), Counter()
    for index, (point, label) in enumerate(gt.items()):
        if point in pred:
            common[label, pred[point]] += 1
        else:
            missing[label] += 1
        if index % 8192 == 0:
            budget.check()
    for point, label in pred.items():
        if point not in gt:
            extra[label] += 1
    g, p, c = len(gt), len(pred), sum(common.values())
    matched = sum(n for (a, b), n in common.items() if a == b)
    assert g == c + sum(missing.values()) and p == c + sum(extra.values())
    # Never allocate an unbounded label-product matrix merely to reject it later.
    counts = ([[common[a, b] for b in pl] for a in gl]
              if len(gl) <= MAX_LABELS and len(pl) <= MAX_LABELS else None)
    strict = {'occupancy_iou': c / (g + p - c) if g + p - c else 1.0,
              'conditional_exact_material_accuracy': matched / c if c else None,
              'gt_material_recall': matched / g if g else None,
              'material_aware_iou': matched / (g + p - matched) if g + p else 1.0,
              'null_reasons': {key: reason for key, reason, missing_denominator in [
                  ('conditional_exact_material_accuracy', 'NO_COMMON_OCCUPANCY', c == 0),
                  ('gt_material_recall', 'EMPTY_GT', g == 0)] if missing_denominator}}
    try:
        if counts is None:
            raise MaterialResourceLimit('LABEL_LIMIT')
        bounds = mapping_upper_bounds(gl, pl, counts, g=g, p=p, original_matches=matched, budget=budget)
    except (MaterialResourceLimit, MemoryError, ImportError) as exc:
        reason = exc.reason if isinstance(exc, MaterialResourceLimit) else type(exc).__name__
        bounds = {kind: {'status': 'NOT_COMPUTED_RESOURCE_OR_DEPENDENCY', 'reason': reason,
                         'mapping': None, 'maximum_exact_matches': None}
                  for kind in ('one_to_one', 'many_to_one')}
    if reference_actions is None or predicted_actions is None:
        action_audit = {'status': 'NOT_VERIFIED', 'reason': 'REFERENCE_OR_PREDICTED_ACTIONS_NOT_SUPPLIED'}
    else:
        try:
            action_audit = audit_reference_actions(reference_actions, predicted_actions, limits=limits, budget=budget)
        except (MaterialResourceLimit, MemoryError) as exc:
            action_audit = {'status': 'NOT_VERIFIED_RESOURCE_LIMIT',
                            'reason': exc.reason if isinstance(exc, MaterialResourceLimit) else 'MemoryError'}
    return {'protocol_id': PROTOCOL_ID, 'analysis_label': 'POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC',
            'g': g, 'p': p, 'C': c, 'M': matched, 'common_material_mismatches': c - matched,
            'missing_occupied_count': g - c, 'extra_occupied_count': p - c,
            'strict': strict,
            'confusion': {'gt_labels': gl, 'pred_labels': pl, 'common_counts': counts,
                          'common_matrix_status': 'COMPLETE' if counts is not None else 'NOT_MATERIALIZED_LABEL_LIMIT',
                          'common_nonzero': [{'gt_material_id': a, 'pred_material_id': b, 'count': n}
                                             for (a, b), n in sorted(common.items())],
                          'missing_by_gt_label': [{'gt_material_id': a, 'pred_material_id': None,
                                                   'category': 'missing_geometry', 'count': missing[a]} for a in gl],
                          'extra_by_pred_label': [{'gt_material_id': None, 'pred_material_id': b,
                                                  'category': 'extra_geometry', 'count': extra[b]} for b in pl],
                          'missing_extra_are_distinct_from_label_assignment': True},
            **bounds, 'reference_action_audit': action_audit,
            'provenance': {'implementation_sha256': implementation_sha256(),
                           'gt_world_grid_sha256': _grid_hash(gt),
                           'pred_world_grid_sha256': _grid_hash(pred_world_dict),
                           'saved_shift_xyz': list(shift), 'shift_sha256': _hash(list(shift)),
                           'source_binding': copy.deepcopy(source_binding),
                           'source_binding_verified_by_caller_only': True,
                           'geometry_optimized': False, 'prediction_artifact_modified': False,
                           'model_output_success_reclassified': False,
                           'research_api_requests_sent': 0, 'api_key_queries': 0}}
