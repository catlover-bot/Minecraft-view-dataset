"""The twelve fixed Goal014 estimands; pure arithmetic, no file/model access.

The caller verifies source/feature/outcome bindings and the committed plan before
calling this module. Original Goal013A scores are never replaced by replay scores.
The gallery is fixed for Q-A; only query-building uncertainty is bootstrapped.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

from ..goal013a.associations import (
    BUILDERS, PROVIDERS, REPLICATES, SCORE_ID, SEED, bootstrap_indices,
    bootstrap_interval, finite, require, spearman,
)


def _key(row):
    return row['scene_id'], row['description_provider']


def _index(rows, expected, label):
    indexed = {_key(row): row for row in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected,
            label + ' must preserve the complete planned frame without duplicate rows')
    return indexed


def _eligible(row):
    if row.get('score_status') != 'COMPUTATION_VALID':
        require(row.get('score') is None, 'invalid original scores must remain null')
        return False
    require(finite(row.get('score')), 'valid original score must be finite')
    return row.get('token_coverage_status') == 'VERIFIED'


def _truncated(row):
    flag = row.get('token_diagnostics', {}).get('truncated')
    require(type(flag) is bool, 'original boolean truncated flag is required')
    length = row.get('summary_token_count_untruncated')
    require(type(length) is int and length > 0, 'original positive token length is required')
    return flag


def _outcome(row, builder, field):
    outcome = row['outcomes'][builder]
    value = outcome.get(field)
    if value is None:
        return None
    require(finite(value) and 0 <= value <= 1 and outcome.get('status') == 'success',
            'finite saved outcome must be a valid success scalar')
    return value


def _mask(rows, scenes):
    keys = sorted(_key(row) for row in rows)
    serialized = json.dumps(keys, ensure_ascii=True, separators=(',', ':')).encode()
    return {
        'paired_description_rows': len(keys),
        'contributing_buildings': len({key[0] for key in keys}),
        'planned_buildings': len(scenes),
        'planned_description_rows': len(scenes) * len(PROVIDERS),
        'row_ids': [list(key) for key in keys],
        'row_ids_sha256': hashlib.sha256(serialized).hexdigest(),
        'provider_denominators': {
            provider: {'paired_description_rows': sum(key[1] == provider for key in keys),
                       'contributing_buildings': len({key[0] for key in keys if key[1] == provider})}
            for provider in PROVIDERS},
    }


def _result(point, sampled, contributed):
    return {'estimate': point,
            'estimate_status': 'DEFINED' if point is not None else 'UNDEFINED_TOO_FEW_ROWS_OR_CONSTANT_RANKS',
            **bootstrap_interval(point, sampled, contributed)}


def _by_scene(rows, scenes):
    indexed = {scene: [] for scene in scenes}
    for row in rows:
        indexed[row['scene_id']].append(row)
    return indexed


def _sample(by_scene, scenes, draw):
    # Repeated building indices deliberately duplicate every eligible row.
    return [row for index in draw for row in by_scene[scenes[int(index)]]]


def _qa_result(rows, scenes, draws, field):
    valid = [row for row in rows if row[field] is not None]
    by_scene = _by_scene(valid, scenes)
    values = [float(np.mean([row[field] for row in by_scene[scene]]))
              if by_scene[scene] else None for scene in scenes]
    present = [value for value in values if value is not None]
    point = float(np.mean(present)) if present else None
    sampled = []
    for draw in draws:
        current = [values[int(index)] for index in draw if values[int(index)] is not None]
        sampled.append(float(np.mean(current)) if current else None)
    result = {'question': 'Q-A', 'estimand': field,
              'weighting': 'mean of available summaries within building, then equal building mean',
              'uncertainty_scope': 'query buildings conditional on the fixed 60-scene candidate gallery',
              'reference_value_not_pass_threshold': 0.0 if field == 'D_i' else 0.5,
              'per_building': [{'scene_id': scene, 'estimate': value,
                                'eligible_summary_rows': len(by_scene[scene])}
                               for scene, value in zip(scenes, values)],
              **_mask(valid, scenes), **_result(point, sampled, len(present))}
    if point is None:
        result['estimate_status'] = 'UNDEFINED_NO_ELIGIBLE_QUERY_BUILDINGS'
    return result


def _paired_correlations(rows, scenes, draws):
    """Both correlations and their paired difference use identical rows/draws."""
    def calculate(selected):
        score = [row['score'] for row in selected]
        iou = spearman(score, [row['iou'] for row in selected])
        surface = spearman(score, [row['surface'] for row in selected])
        delta = surface - iou if iou is not None and surface is not None else None
        return iou, surface, delta
    points = calculate(rows)
    by_scene = _by_scene(rows, scenes)
    samples = [[], [], []]
    for draw in draws:
        for destination, value in zip(samples, calculate(_sample(by_scene, scenes, draw))):
            destination.append(value)
    return points, samples


def _pooled(rows, scenes, draws):
    def calculate(selected):
        return spearman([row['score'] for row in selected], [row['iou'] for row in selected])
    by_scene = _by_scene(rows, scenes)
    return calculate(rows), [calculate(_sample(by_scene, scenes, draw)) for draw in draws]


def _token_description(rows):
    lengths = [row['summary_token_count_untruncated'] for row in rows]
    return {'description_rows': len(rows),
            'provider_counts': {provider: sum(row['description_provider'] == provider for row in rows)
                                for provider in PROVIDERS},
            'untruncated_summary_token_length': {
                'min': min(lengths) if lengths else None,
                'median': float(np.median(lengths)) if lengths else None,
                'max': max(lengths) if lengths else None}}


def analyze(scored_rows, outcome_rows, qa_rows, planned_scene_ids, *, fixture=False):
    """Return exactly Q-A 2, Q-B 6, Q-C 4 fixed estimates with all 2000 draws.

    All three row collections retain the full planned scene/provider frame.
    Scores have Goal013A's schema, including token_diagnostics.truncated.
    Outcome rows contain outcomes[builder] with status, aligned_occupancy_iou,
    aligned_surface_f1_at_1. QA rows contain D_i/W_i (both null when unavailable).
    fixture=True only allows smaller synthetic frames/counts; inference constants
    and missing/tie/CI rules remain fixed. No input object is mutated.
    """
    scenes = sorted(planned_scene_ids)
    require(scenes and len(scenes) == len(set(scenes)), 'unique planned scene frame required')
    require(fixture or len(scenes) == 60, 'production frame requires all 60 planned buildings')
    expected = {(scene, provider) for scene in scenes for provider in PROVIDERS}
    scores = _index(scored_rows, expected, 'score rows')
    outcomes = _index(outcome_rows, expected, 'outcome rows')
    qa = _index(qa_rows, expected, 'Q-A rows')
    eligible = []
    for key in sorted(expected):
        score, outcome, diagnostic = scores[key], outcomes[key], qa[key]
        require(score.get('score_id') == SCORE_ID, 'fixed original Goal013A score ID required')
        require(set(outcome.get('outcomes', {})) == set(BUILDERS), 'two separate Builder outcomes required')
        for builder in BUILDERS:
            for field in ('aligned_occupancy_iou', 'aligned_surface_f1_at_1'):
                _outcome(outcome, builder, field)
        is_eligible = _eligible(score)
        if is_eligible:
            _truncated(score)
            eligible.append(score)
        d, w = diagnostic.get('D_i'), diagnostic.get('W_i')
        require((d is None) == (w is None), 'Q-A D/W must have the same eligible rows')
        if d is not None:
            require(is_eligible and finite(d) and finite(w) and 0 <= w <= 1,
                    'Q-A requires eligible original summary and finite D/W')
        else:
            require(not is_eligible or diagnostic.get('null_reason'), 'missing Q-A result requires an explicit reason')
    groups = {flag: [row for row in eligible if _truncated(row) is flag] for flag in (False, True)}
    if not fixture:
        require(len(eligible) == 178 and len(groups[False]) == 119 and len(groups[True]) == 59,
                'fixed original 178/119/59 summary/truncation counts differ')
        require(Counter(row['description_provider'] for row in groups[True]) ==
                {'openai': 38, 'anthropic': 14, 'gemini': 7}, 'original provider truncation counts differ')
    draws = bootstrap_indices(scenes)
    result = {
        'schema': 'goal014_fixed_statistics_v1', 'status': 'POST_HOC_EXPLORATORY',
        'score_id': SCORE_ID, 'planned_scene_ids': scenes, 'planned_buildings': len(scenes),
        'planned_description_rows': len(expected), 'eligible_original_score_rows': len(eligible),
        'maximum_estimands': 12, 'inference': {
            'rng': 'NumPy Generator(PCG64)', 'seed': SEED, 'replicates': REPLICATES,
            'minimum_finite_draws_for_ci': 1900, 'minimum_contributing_buildings_for_ci': 2,
            'quantile_method': 'linear / Hyndman-Fan type 7', 'same_draws_for_all_estimands': True,
            'index_matrix_sha256_little_endian_int64': hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
            'numpy_version': np.__version__, 'p_values_computed': False,
            'resampled_frame': 'all planned buildings, including those without eligible rows'},
        'q_a': [], 'q_b': [], 'q_c': [],
        'truncation_groups': {str(flag).lower(): _token_description(groups[flag]) for flag in (False, True)},
    }
    qa_ordered = [qa[key] for key in sorted(expected)]
    for field in ('D_i', 'W_i'):
        result['q_a'].append(_qa_result(qa_ordered, scenes, draws, field))
    for builder in BUILDERS:
        original_valid, common = [], []
        for row in eligible:
            outcome = outcomes[_key(row)]
            iou = _outcome(outcome, builder, 'aligned_occupancy_iou')
            surface = _outcome(outcome, builder, 'aligned_surface_f1_at_1')
            if iou is not None:
                original_valid.append(row)
                if surface is not None:
                    common.append({'scene_id': row['scene_id'], 'description_provider': row['description_provider'],
                                   'score': row['score'], 'iou': iou, 'surface': surface})
        mask, old_mask = _mask(common, scenes), _mask(original_valid, scenes)
        points, samples = _paired_correlations(common, scenes, draws)
        for name, point, sampled in zip(('rho_iou', 'rho_surface', 'delta_rho'), points, samples):
            result['q_b'].append({'question': 'Q-B', 'builder': builder, 'estimand': name,
                'weighting': 'pooled description rows; building clusters for interval only',
                'common_valid_set': True, 'common_set_equals_original_iou_set': mask['row_ids'] == old_mask['row_ids'],
                'original_valid_iou_rows': old_mask['paired_description_rows'],
                'original_valid_iou_row_ids': old_mask['row_ids'],
                'original_valid_iou_row_ids_sha256': old_mask['row_ids_sha256'],
                **mask, **_result(point, sampled, mask['contributing_buildings'])})
        for flag in (False, True):
            paired = []
            source_rows = []
            for row in groups[flag]:
                iou = _outcome(outcomes[_key(row)], builder, 'aligned_occupancy_iou')
                if iou is not None:
                    paired.append({'scene_id': row['scene_id'], 'description_provider': row['description_provider'],
                                   'score': row['score'], 'iou': iou})
                    source_rows.append(row)
            mask = _mask(paired, scenes)
            point, sampled = _pooled(paired, scenes, draws)
            result['q_c'].append({'question': 'Q-C', 'builder': builder, 'estimand': 'rho_iou',
                'original_truncated': flag, 'sensitivity_only': True,
                'weighting': 'pooled description rows; building clusters for interval only',
                'subset_description': _token_description(source_rows),
                **mask, **_result(point, sampled, mask['contributing_buildings'])})
    require(sum(len(result[key]) for key in ('q_a', 'q_b', 'q_c')) == 12, 'exactly twelve fixed estimands required')
    return result
