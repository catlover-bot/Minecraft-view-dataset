"""Frozen exploratory CPU proxy associations; no generation or credential I/O."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

PROTOCOL_ID = 'goal008_cpu_proxy_associations_v1'
SEED = 20261016
REPLICATES = 2000
MIN_VALID_REPLICATES = 1900
FEATURES = ('strict_material_f1', 'coarse_material_f1', 'dimension_accuracy', 'completeness', 'composite')
PROVIDERS = ('openai', 'anthropic', 'gemini')
BUILDERS = ('openai', 'anthropic')
TARGETS = (*BUILDERS, 'both_builders_mean')
POLICIES = ('valid_only', 'bounded_iou_failure0')
ASSOCIATIONS = ('pooled_spearman', 'within_building_centered_pearson', 'within_building_midrank_association')
DEFINITIVE_FAILURES = {'parse_failure', 'validation_failure', 'execution_failure', 'truncation'}
HASH_FIELDS = ('input_sha256', 'source_sha256', 'gt_bbox_sha256', 'gt_voxels_sha256')
DEFINITION = Path(__file__).resolve().parents[3] / 'reports/server_only/goal008/stats_definition.md'


def definition_sha256():
    return hashlib.sha256(DEFINITION.read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _gate(gate, digest):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in {'fixture', 'verified_sources'}):
        raise ValueError('passing explicit input-kind audit gate required')
    if not isinstance(digest, str) or digest != definition_sha256() or gate.get('definition_sha256') != digest:
        raise ValueError('frozen statistics definition SHA256 mismatch')
    if gate.get('phase0_binding') != 'PASS' or gate.get('definitions_frozen') is not True:
        raise ValueError('Phase0 binding and frozen definitions required before analysis')
    if gate['input_kind'] == 'verified_sources' and gate.get('matched_description_count') != 180:
        raise ValueError('all180 source bindings required for actual proxy analysis')


def _number(value, label):
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(label + ' must be a finite number in [0,1]')
    return float(value)


def _outcome(row):
    status = row['status']
    if type(row.get('usage_known')) is not bool or type(row.get('completed_request_outcome')) is not bool:
        raise ValueError('explicit completed-response and usage booleans required')
    known = row['usage_known'] and row['completed_request_outcome']
    if status == 'success':
        metrics = row.get('metrics')
        if not known or not isinstance(metrics, dict) or metrics.get('metric_version') != 'server_v2.2_strict_translation':
            raise ValueError('known completed strict-metric success required')
        valid = _number(metrics.get('aligned', {}).get('occupancy_iou'), 'O aligned occupancy IoU')
    else:
        if row.get('metrics') is not None:
            raise ValueError('non-success outcome metrics must remain null')
        valid = None
    if row.get('http_status_code', row.get('http_status')) in (401, 520) and known:
        raise ValueError('operational unknown cannot become identified completed failure0')
    failure0 = status == 'failed' and known and row.get('failure_kind') in DEFINITIVE_FAILURES
    if status not in {'success', 'failed', 'operational_unmeasurable', 'ALIGNMENT_RESOURCE_LIMIT', 'GT_MISSING'}:
        raise ValueError('unexpected final O observation status')
    return {'global_schedule_index': row['global_schedule_index'], 'attempt_id': row['attempt_id'],
            'status': status, 'failure_kind': row.get('failure_kind'), 'usage_known': row['usage_known'],
            'completed_request_outcome': row['completed_request_outcome'], 'valid_only': valid,
            'bounded_iou_failure0': valid if valid is not None else 0.0 if failure0 else None,
            'failure0_applied': failure0}


def join_descriptions(score_rows, slots, *, fixed_design):
    """Exact source-bound O join; duplicated Builder outcomes are never pooled."""
    scores, seen_inputs = {}, set()
    for row in score_rows:
        key = (row.get('scene_id'), row.get('description_provider'))
        if (not isinstance(key[0], str) or not key[0] or key[1] not in PROVIDERS or key in scores
                or not isinstance(row.get('input_id'), str) or not row['input_id'] or row['input_id'] in seen_inputs):
            raise ValueError('unique scene/description provider and input identity required')
        for name in HASH_FIELDS:
            if not isinstance(row.get(name), str) or not re.fullmatch('[0-9a-f]{64}', row[name]):
                raise ValueError('explicit source hash required: ' + name)
        values = row.get('scores', {})
        if set(values) != set(FEATURES):
            raise ValueError('exact five frozen proxy features required')
        for name in FEATURES:
            if values[name] is None:
                if not isinstance(row.get('missing_reasons', {}).get(name), str) or not row['missing_reasons'][name]:
                    raise ValueError('null proxy requires an explicit missing reason')
            else:
                _number(values[name], name)
        scores[key] = row
        seen_inputs.add(row['input_id'])
    if not scores:
        raise ValueError('nonempty description cohort required')
    if fixed_design:
        scenes = {s for s, p in scores}
        if len(scores) != 180 or len(scenes) != 60 or set(scores) != {(s, p) for s in scenes for p in PROVIDERS}:
            raise ValueError('actual score cohort must be60 buildings by3 descriptions')
        if (len(slots) != 960 or {r['global_schedule_index'] for r in slots} != set(range(960))
                or Counter(r['status'] for r in slots) != {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}):
            raise ValueError('frozen Goal007960 slot status inventory required')
    outcomes = {}
    for row in slots:
        if row['condition'] != 'O':
            continue
        key = (row['scene_id'], row['description_model'])
        full = (*key, row['provider'])
        if key not in scores or row['provider'] not in BUILDERS or full in outcomes:
            raise ValueError('unexpected or duplicated O Builder observation')
        source = scores[key]
        if row['input_id'] != source['input_id'] or any(row.get(name) != source[name] for name in HASH_FIELDS):
            raise ValueError('O outcome source binding mismatch; no hash aliases permitted')
        outcomes[full] = _outcome(row)
    if set(outcomes) != {(*key, builder) for key in scores for builder in BUILDERS}:
        raise ValueError('each description requires exactly both Builder outcome rows')
    return [{**{k: source[k] for k in ('scene_id', 'description_provider', 'input_id', *HASH_FIELDS)},
             'scores': dict(source['scores']), 'missing_reasons': dict(source.get('missing_reasons', {})),
             'builders': {builder: outcomes[(*key, builder)] for builder in BUILDERS}}
            for key, source in sorted(scores.items())]


def _midranks(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values.copy()
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return ranks[inverse]


def _weighted_midranks(values, weights):
    """Midranks of explicitly duplicated observations without materializing copies."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(values[order])) + 1]
    grouped_weights = np.add.reduceat(weights[:, order], starts, axis=1)
    ranks = np.cumsum(grouped_weights, axis=1) - (grouped_weights - 1) / 2
    inverse = np.empty(len(values), dtype=int)
    inverse[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(values)]))
    return ranks[:, inverse]


def _weighted_correlations(x, y, weights):
    """One Pearson coefficient per nonnegative integer multiplicity row."""
    x, y, weights = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(weights, dtype=float)
    x = np.broadcast_to(x, weights.shape); y = np.broadcast_to(y, weights.shape)
    total = weights.sum(axis=1)
    positive = weights > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = x - (x * weights).sum(axis=1)[:, None] / total[:, None]
        dy = y - (y * weights).sum(axis=1)[:, None] / total[:, None]
        vx, vy = (weights * dx * dx).sum(axis=1), (weights * dy * dy).sum(axis=1)
        coefficients = (weights * dx * dy).sum(axis=1) / np.sqrt(vx * vy)
    constant_x = np.max(np.where(positive, x, -np.inf), axis=1) == np.min(np.where(positive, x, np.inf), axis=1)
    constant_y = np.max(np.where(positive, y, -np.inf), axis=1) == np.min(np.where(positive, y, np.inf), axis=1)
    coefficients[(total < 2) | constant_x | constant_y | (vx <= 0) | (vy <= 0)] = np.nan
    return np.clip(coefficients, -1, 1)


def _draw_counts(buildings):
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = rng.integers(0, buildings, size=(REPLICATES, buildings))
    counts = np.zeros((REPLICATES, buildings), dtype=np.int64)
    np.add.at(counts, (np.arange(REPLICATES)[:, None], draws), 1)
    return counts


def _ties(values):
    if not len(values):
        return {'unique_values': 0, 'tie_groups': 0, 'tied_observations': 0, 'constant': None}
    unique, counts = np.unique(values, return_counts=True)
    return {'unique_values': len(unique), 'tie_groups': int(np.sum(counts > 1)),
            'tied_observations': int(np.sum(counts[counts > 1])), 'constant': len(unique) == 1}


def _target(row, target, policy):
    if target in BUILDERS:
        return row['builders'][target][policy]
    values = [row['builders'][builder][policy] for builder in BUILDERS]
    return sum(values) / 2 if all(v is not None for v in values) else None


def _vectors(selected, feature, target, policy, association, scene_index):
    complete, missing_proxy, missing_outcome, missing_both = [], 0, 0, 0
    known_failures, unknown_usage, converted = 0, 0, 0
    for row in selected:
        x, y = row['scores'][feature], _target(row, target, policy)
        required = [row['builders'][builder] for builder in ((target,) if target in BUILDERS else BUILDERS)]
        known_failures += int(any(outcome['failure0_applied'] for outcome in required))
        unknown_usage += int(any(not outcome['usage_known'] for outcome in required))
        converted += int(policy == 'bounded_iou_failure0' and y is not None
                         and any(outcome['failure0_applied'] for outcome in required))
        missing_proxy += int(x is None); missing_outcome += int(y is None)
        missing_both += int(x is None and y is None)
        if x is not None and y is not None:
            complete.append((scene_index[row['scene_id']], float(x), float(y)))
    clustered = {}
    for scene, x, y in complete:
        clustered.setdefault(scene, []).append((x, y))
    vectors, singleton_buildings, constant_proxy_buildings, constant_outcome_buildings = [], 0, 0, 0
    for scene, pairs in clustered.items():
        x, y = np.asarray(pairs, dtype=float).T
        singleton_buildings += int(len(pairs) < 2)
        if len(pairs) >= 2:
            constant_proxy_buildings += int(np.all(x == x[0]))
            constant_outcome_buildings += int(np.all(y == y[0]))
        if association != 'pooled_spearman':
            if len(pairs) < 2:
                continue
            if association == 'within_building_midrank_association':
                x, y = _midranks(x), _midranks(y)
            x = np.zeros_like(x) if np.all(x == x[0]) else x - np.mean(x)
            y = np.zeros_like(y) if np.all(y == y[0]) else y - np.mean(y)
        vectors.extend((scene, a, b) for a, b in zip(x, y))
    clusters = np.array([v[0] for v in vectors], dtype=int)
    x, y = np.array([v[1] for v in vectors]), np.array([v[2] for v in vectors])
    counts = {'planned_descriptions': len(selected), 'planned_buildings': len({r['scene_id'] for r in selected}),
              'missing_proxy': missing_proxy, 'missing_outcome': missing_outcome, 'missing_both': missing_both,
              'missing_proxy_only': missing_proxy - missing_both, 'missing_outcome_only': missing_outcome - missing_both,
              'known_model_failure_descriptions': known_failures, 'unknown_usage_descriptions': unknown_usage,
              'failure0_converted_descriptions': converted,
              'complete_pairs': len(complete), 'complete_pair_buildings': len(clustered),
              'analysis_pairs': len(vectors), 'analysis_buildings': len(set(clusters.tolist())),
              'singleton_buildings': singleton_buildings,
              'excluded_singleton_buildings': singleton_buildings if association != 'pooled_spearman' else 0,
              'constant_proxy_buildings': constant_proxy_buildings, 'constant_outcome_buildings': constant_outcome_buildings,
              'proxy_ties': _ties(x), 'outcome_ties': _ties(y)}
    return x, y, clusters, counts


def _association(selected, feature, target, policy, association, scope, scene_index, draws):
    x, y, clusters, counts = _vectors(selected, feature, target, policy, association, scene_index)
    record = {'id': ':'.join((scope or 'all', policy, target, feature, association)), 'feature': feature,
              'description_provider': scope, 'scope': 'all_description_providers' if scope is None else scope,
              'outcome_target': target, 'outcome_policy': policy, 'association': association,
              'coefficient': None, 'ci95': None, 'counts': counts, 'bootstrap_attempted': 0,
              'bootstrap_valid': 0, 'bootstrap_invalid': 0, 'bootstrap_planned': REPLICATES,
              'independent_unit': 'building', 'observation_unit': 'source-bound description',
              'weighting': 'equal eligible description pairs; building-cluster uncertainty',
              'exploratory': True, 'p_values_computed': False}
    if scope is not None and association != 'pooled_spearman':
        return dict(record, status='NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING', ci_status='NOT_ESTIMABLE')
    if len(x) < 2:
        return dict(record, status='INSUFFICIENT_PAIRS', ci_status='NOT_ESTIMABLE')
    constant_x, constant_y = np.all(x == x[0]), np.all(y == y[0])
    if constant_x or constant_y:
        reason = 'CONSTANT_BOTH' if constant_x and constant_y else 'CONSTANT_PROXY' if constant_x else 'CONSTANT_OUTCOME'
        return dict(record, status=reason, ci_status='NOT_ESTIMABLE')
    point_x, point_y = (_midranks(x), _midranks(y)) if association == 'pooled_spearman' else (x, y)
    coefficient = _weighted_correlations(point_x, point_y, np.ones((1, len(x))))[0]
    if not np.isfinite(coefficient):
        return dict(record, status='NUMERICALLY_UNDEFINED', ci_status='NOT_ESTIMABLE')
    record['coefficient'] = float(coefficient)
    if counts['analysis_buildings'] < 2:
        return dict(record, status='ESTIMATE_WITHOUT_CLUSTER_CI', ci_status='INSUFFICIENT_BUILDINGS')
    weights = draws[:, clusters]
    draw_x, draw_y = (_weighted_midranks(x, weights), _weighted_midranks(y, weights)) if association == 'pooled_spearman' else (x, y)
    coefficients = _weighted_correlations(draw_x, draw_y, weights)
    finite = coefficients[np.isfinite(coefficients)]
    record.update(bootstrap_attempted=REPLICATES, bootstrap_valid=len(finite), bootstrap_invalid=REPLICATES-len(finite))
    if len(finite) < MIN_VALID_REPLICATES:
        return dict(record, status='ESTIMATE_WITHOUT_CLUSTER_CI', ci_status='UNSTABLE_BOOTSTRAP')
    return dict(record, status='ESTIMABLE', ci_status='PERCENTILE_95',
                ci95=np.quantile(finite, [.025, .975], method='linear').tolist())


def analyze_correlations(score_rows, slots, *, audit_gate, definition_sha256):
    """Analyze only fixed, source-bound scores after Phase0 and definition gates."""
    _gate(audit_gate, definition_sha256)
    joined = join_descriptions(score_rows, slots, fixed_design=audit_gate['input_kind'] == 'verified_sources')
    scenes = sorted({row['scene_id'] for row in joined})
    draw_cache = {}
    associations = []
    for scope in (None, *PROVIDERS):
        selected = [row for row in joined if scope is None or row['description_provider'] == scope]
        scope_scenes = tuple(sorted({row['scene_id'] for row in selected}))
        scene_index = {scene: index for index, scene in enumerate(scope_scenes)}
        if scope_scenes not in draw_cache:
            draw_cache[scope_scenes] = (_draw_counts(len(scope_scenes)) if scope_scenes
                                        else np.zeros((REPLICATES, 0), dtype=np.int64))
        draws = draw_cache[scope_scenes]
        for policy in POLICIES:
            for target in TARGETS:
                for feature in FEATURES:
                    for association in ASSOCIATIONS:
                        associations.append(_association(selected, feature, target, policy, association, scope, scene_index, draws))
    assert len(associations) == 360
    return {'protocol_id': PROTOCOL_ID, 'analysis_kind': 'new_exploratory_cpu_attribute_fidelity_proxy_associations',
            'conventional_rq2_status': 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES',
            'input_kind': audit_gate['input_kind'], 'definition_sha256': definition_sha256,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inference': {'seed': SEED, 'generator': 'NumPy PCG64', 'cluster_bootstrap_replicates': REPLICATES,
                          'minimum_valid_replicates': MIN_VALID_REPLICATES, 'interval': '95% percentile, linear quantiles',
                          'cluster_sampling_frame': 'all planned buildings; duplicate draws are distinct cluster copies',
                          'shared_draw_matrix': True, 'p_values': False, 'holm_family_changes': False},
            'source_binding': {'matched_descriptions': len(joined), 'buildings': len(scenes),
                               'builder_observations': len(joined)*2, 'independent_unit': 'building',
                               'joined_metadata_sha256': _digest(joined)},
            'joined_descriptions': joined, 'associations': associations,
            'status_counts': dict(sorted(Counter(row['status'] for row in associations).items())),
            'limitations': ['New CPU proxy, not conventional caption quality or human judgment.',
                            'Shared GT, exploratory multiplicity and valid selection prevent causal interpretation.',
                            'Builder outcomes repeat each description and are never pooled as independent observations.',
                            'Within-building provider strata are nonestimable with one description per building.'],
            'research_api_requests_sent': 0, 'api_key_queries': 0, 'human_experiments': 0,
            'goal007_confirmatory_tests_reexecuted': False}
