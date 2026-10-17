"""Fixed Goal013 Phase 2 arithmetic; no filesystem, model or network operations.

Production callers must obtain rows/gate from outcome_binding only after the
score artifact is frozen. NumPy is the sole numerical dependency; no scipy.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import math

import numpy as np

PROVIDERS = ('openai', 'anthropic', 'gemini')
BUILDERS = ('openai', 'anthropic')
POLICIES = ('valid_only', 'known_failure0')
SEED = 20261017
REPLICATES = 2000
MIN_VALID_DRAWS = 1900
SCORE_ID = 'MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def average_ranks(values):
    """One-based average ranks; equality is exact, before presentation rounding."""
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and np.isfinite(values).all(), 'finite rank vector required')
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return result


def spearman(x, y):
    """Pearson of average ranks. No p-value, epsilon ties or implicit deletion."""
    require(len(x) == len(y), 'paired vectors differ')
    if len(x) < 2:
        return None
    a, b = average_ranks(x), average_ranks(y)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator == 0 or not math.isfinite(denominator):
        return None
    value = float(np.dot(a, b) / denominator)
    return value if math.isfinite(value) else None


def bootstrap_indices(scene_ids):
    require(scene_ids == sorted(set(scene_ids)) and scene_ids, 'sorted unique planned frame required')
    return np.random.Generator(np.random.PCG64(SEED)).integers(
        0, len(scene_ids), size=(REPLICATES, len(scene_ids)), dtype=np.int64)


def bootstrap_interval(point, values, contributing_buildings):
    require(len(values) == REPLICATES, 'exactly 2000 bootstrap draws required')
    observed = [v for v in values if v is not None]
    require(all(finite(v) for v in observed), 'invalid bootstrap value')
    enough = point is not None and contributing_buildings >= 2 and len(observed) >= MIN_VALID_DRAWS
    return {
        'ci95': [float(v) for v in np.quantile(observed, [.025, .975], method='linear')] if enough else None,
        'ci_status': 'EXPLORATORY_CONDITIONAL_ON_FINITE_DRAWS' if enough else 'INSUFFICIENT_BOOTSTRAP_VALIDITY',
        'bootstrap_finite_draws': len(observed), 'bootstrap_undefined_draws': REPLICATES - len(observed),
        'bootstrap_draws': values,
    }


def _score_reason(row):
    if row.get('score_status') != 'COMPUTATION_VALID':
        return 'SCORE_NOT_COMPUTATION_VALID'
    require(finite(row.get('score')), 'COMPUTATION_VALID score must be finite')
    if row.get('token_coverage_status') != 'VERIFIED':
        return 'TOKEN_COVERAGE_NOT_VERIFIED'
    return None


def _outcome(row, builder, policy):
    outcome = row['outcomes'][builder]
    value = outcome.get('aligned_occupancy_iou')
    if value is not None:
        require(finite(value) and 0 <= value <= 1 and outcome['status'] == 'success', 'invalid saved IoU')
        return value, 0
    if policy == 'known_failure0' and outcome['known_failure0_eligible'] is True:
        return 0.0, 1
    return None, 0


def _target(row, target, policy):
    if target in BUILDERS:
        return _outcome(row, target, policy)
    require(target == 'both_builders_mean', 'unknown target')
    values = [_outcome(row, builder, policy) for builder in BUILDERS]
    if any(v is None for v, _ in values):
        return None, 0
    return sum(v for v, _ in values) / 2.0, sum(n for _, n in values)


def _pairs(rows, target, policy):
    pairs = []
    for row in rows:
        if _score_reason(row) is not None:
            continue
        value, imputed = _target(row, target, policy)
        if value is not None:
            pairs.append((row['scene_id'], row['description_provider'], row['score'], value, imputed))
    return pairs


def _pooled_result(pairs, scene_ids, draws):
    by_scene = {scene: [] for scene in scene_ids}
    for pair in pairs:
        by_scene[pair[0]].append(pair)
    point = spearman([r[2] for r in pairs], [r[3] for r in pairs])
    sampled_values = []
    for draw in draws:
        # Concatenation keeps duplicate scene occurrences. Ranks are recomputed.
        sampled = [row for index in draw for row in by_scene[scene_ids[int(index)]]]
        sampled_values.append(spearman([r[2] for r in sampled], [r[3] for r in sampled]))
    contributed = sum(bool(values) for values in by_scene.values())
    return {
        'estimate': point, 'estimate_status': 'DEFINED' if point is not None else 'UNDEFINED_TOO_FEW_ROWS_OR_CONSTANT_RANKS',
        'paired_description_rows': len(pairs), 'contributing_buildings': contributed,
        'failure0_imputed_description_rows': sum(r[4] > 0 for r in pairs),
        'failure0_imputed_builder_slots': sum(r[4] for r in pairs),
        'provider_denominators': {
            p: {'paired_description_rows': sum(r[1] == p for r in pairs),
                'contributing_buildings': len({r[0] for r in pairs if r[1] == p})}
            for p in PROVIDERS},
        **bootstrap_interval(point, sampled_values, contributed),
    }


def _concordance_result(rows, scene_ids, draws, builder, policy):
    indexed = {(r['scene_id'], r['description_provider']): r for r in rows}
    total = Counter({k: 0 for k in ('missing_pairs', 'score_only_tie_pairs', 'outcome_only_tie_pairs',
                                  'both_tie_pairs', 'comparable_pairs', 'concordant_pairs', 'discordant_pairs')})
    buildings = []
    for scene in scene_ids:
        counts = Counter({key: 0 for key in total})
        for a, b in combinations(PROVIDERS, 2):
            ra, rb = indexed[(scene, a)], indexed[(scene, b)]
            ya, _ = _outcome(ra, builder, policy)
            yb, _ = _outcome(rb, builder, policy)
            if _score_reason(ra) is not None or _score_reason(rb) is not None or ya is None or yb is None:
                counts['missing_pairs'] += 1
                continue
            score_tie, outcome_tie = ra['score'] == rb['score'], ya == yb
            if score_tie or outcome_tie:
                key = 'both_tie_pairs' if score_tie and outcome_tie else 'score_only_tie_pairs' if score_tie else 'outcome_only_tie_pairs'
                counts[key] += 1
                continue
            concordant = (ra['score'] > rb['score']) == (ya > yb)
            counts['comparable_pairs'] += 1
            counts['concordant_pairs' if concordant else 'discordant_pairs'] += 1
        estimate = counts['concordant_pairs'] / counts['comparable_pairs'] if counts['comparable_pairs'] else None
        buildings.append({'scene_id': scene, 'estimate': estimate, 'planned_candidate_pairs': 3, **dict(counts)})
        total.update(counts)
    estimates = [r['estimate'] for r in buildings if r['estimate'] is not None]
    point = float(np.mean(estimates)) if estimates else None
    sampled_values = []
    for draw in draws:
        sampled = [buildings[int(index)]['estimate'] for index in draw if buildings[int(index)]['estimate'] is not None]
        sampled_values.append(float(np.mean(sampled)) if sampled else None)
    pairs = _pairs(rows, builder, policy)
    return {
        'estimate': point, 'estimate_status': 'DEFINED' if point is not None else 'NO_COMPARABLE_PAIRS',
        'paired_description_rows': len(pairs), 'contributing_buildings': len(estimates),
        'planned_candidate_pairs': len(scene_ids) * 3, **dict(total),
        'failure0_imputed_description_rows': sum(r[4] > 0 for r in pairs),
        'failure0_imputed_builder_slots': sum(r[4] for r in pairs),
        'per_building': buildings,
        **bootstrap_interval(point, sampled_values, len(estimates)),
    }


def analyze(rows, planned_scene_ids, *, phase2_gate, fixture=False):
    """Calculate only the pre-fixed A/B and four token-length diagnostics.

    fixture=True permits a smaller planned frame, never different inferential
    constants. The production loader provides the phase2_gate receipt.
    """
    require(phase2_gate.get('status') == 'PASS' and phase2_gate.get('score_freeze_verified') is True
            and phase2_gate.get('representativeness_status') == 'PASS'
            and phase2_gate.get('outcome_binding_status') == 'PASS', 'Phase 2 gates must pass before arithmetic')
    scenes = sorted(planned_scene_ids)
    require(scenes and len(scenes) == len(set(scenes)), 'unique planned scene IDs required')
    require(fixture or len(scenes) == 60, 'production planned frame must contain 60 buildings')
    keys = {(r['scene_id'], r['description_provider']) for r in rows}
    require(len(keys) == len(rows) and keys == {(s, p) for s in scenes for p in PROVIDERS}, 'complete planned score frame required')
    for row in rows:
        require(set(row['outcomes']) == set(BUILDERS), 'two separate Builder outcomes required')
        _score_reason(row)
        for builder in BUILDERS:
            require(type(row['outcomes'][builder].get('known_failure0_eligible')) is bool, 'explicit failure0 eligibility required')
            _outcome(row, builder, 'valid_only')
    draws = bootstrap_indices(scenes)
    excluded = Counter(_score_reason(r) for r in rows if _score_reason(r) is not None)
    result = {
        'schema': 'goal013a_fixed_phase2_v1', 'status': 'POST_HOC_EXPLORATORY',
        'score_id': SCORE_ID, 'planned_buildings': len(scenes), 'planned_description_rows': len(rows),
        'score_eligible_description_rows': sum(_score_reason(r) is None for r in rows),
        'score_exclusion_counts': dict(excluded), 'planned_scene_ids': scenes,
        'inference': {'rng': 'NumPy Generator(PCG64)', 'seed': SEED, 'replicates': REPLICATES,
            'minimum_finite_draws_for_ci': MIN_VALID_DRAWS, 'minimum_contributing_buildings_for_ci': 2,
            'quantile_method': 'linear / Hyndman-Fan type 7', 'same_draws_for_all_estimands': True,
            'index_matrix_sha256_little_endian_int64': hashlib.sha256(draws.astype('<i8', copy=False).tobytes()).hexdigest(),
            'numpy_version': np.__version__, 'p_values_computed': False},
        'pooled_spearman': [], 'within_building_concordance': [], 'token_length_diagnostics': [],
    }
    for policy in POLICIES:
        for target in (*BUILDERS, 'both_builders_mean'):
            pairs = _pairs(rows, target, policy)
            result['pooled_spearman'].append({'policy': policy, 'target': target,
                'planned_buildings': len(scenes), 'planned_description_rows': len(rows),
                **_pooled_result(pairs, scenes, draws)})
        for builder in BUILDERS:
            result['within_building_concordance'].append({'policy': policy, 'builder': builder,
                'planned_buildings': len(scenes), 'planned_description_rows': len(rows),
                **_concordance_result(rows, scenes, draws, builder, policy)})
    for provider in ('all', *PROVIDERS):
        selected = [r for r in rows if provider == 'all' or r['description_provider'] == provider]
        pairs = []
        for row in selected:
            if _score_reason(row) is not None:
                continue
            length = row.get('summary_token_count_untruncated')
            require(type(length) is int and length > 0, 'verified token coverage requires positive original token count')
            pairs.append((row['scene_id'], row['description_provider'], row['score'], length, 0))
        result['token_length_diagnostics'].append({'description_provider': provider,
            'diagnostic': 'score_vs_untruncated_summary_token_length',
            'planned_buildings': len(scenes), 'planned_description_rows': len(selected),
            **_pooled_result(pairs, scenes, draws)})
    return result
