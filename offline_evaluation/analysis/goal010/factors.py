"""Post-hoc description-provider contrasts; CPU-only and no p-values."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

import numpy as np

from ..goal007 import statistics as original

LABEL = 'POST_HOC_EXPLORATORY'
PROTOCOL_ID = 'goal010_description_provider_paired_v1'
SEED = 20261017
REPLICATES = 2000
MIN_VALID_REPLICATES = 1900
BUILDERS = ('openai', 'anthropic')
PROVIDERS = ('openai', 'anthropic', 'gemini')
CONDITIONS = ('O', 'J')
METRICS = ('aligned_occupancy_iou', 'aligned_dimension_mae', 'aligned_material_aware_iou')
PAIRS = (('openai', 'anthropic'), ('openai', 'gemini'), ('anthropic', 'gemini'))


def require_gate(gate, specification_sha256):
    if (not isinstance(gate, dict) or gate.get('status') != 'PASS'
            or gate.get('input_kind') not in {'fixture', 'verified_sources'}
            or gate.get('phase0_binding') != 'PASS' or gate.get('definitions_frozen') is not True):
        raise ValueError('passing Phase0 gate and frozen post-hoc specification required')
    if (not isinstance(specification_sha256, str) or not re.fullmatch('[0-9a-f]{64}', specification_sha256)
            or gate.get('specification_sha256') != specification_sha256):
        raise ValueError('post-hoc specification hash mismatch')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def draw_counts(buildings):
    if buildings == 0:
        return np.zeros((REPLICATES, 0), dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = rng.integers(0, buildings, size=(REPLICATES, buildings))
    counts = np.zeros((REPLICATES, buildings), dtype=np.int64)
    np.add.at(counts, (np.arange(REPLICATES)[:, None], draws), 1)
    return counts


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('finite building differences required')
    result = {'mean_delta': float(np.mean(values)) if len(values) else None, 'ci95': None,
              'bootstrap_planned': REPLICATES, 'bootstrap_valid': 0, 'bootstrap_invalid': 0,
              'bootstrap_attempted': 0, 'ci_status': 'INSUFFICIENT_BUILDINGS'}
    if len(values) < 2:
        return dict(result, status='INSUFFICIENT_BUILDINGS')
    bootstraps = (draw_counts(len(values)) * values).sum(axis=1) / len(values)
    finite = bootstraps[np.isfinite(bootstraps)]
    result.update(bootstrap_attempted=REPLICATES, bootstrap_valid=len(finite), bootstrap_invalid=REPLICATES-len(finite))
    if len(finite) < MIN_VALID_REPLICATES:
        return dict(result, status='ESTIMATE_WITHOUT_CLUSTER_CI', ci_status='UNSTABLE_BOOTSTRAP')
    return dict(result, status='ESTIMABLE', ci_status='PERCENTILE_95',
                ci95=np.quantile(finite, [.025, .975], method='linear').tolist())


def _evidence(row):
    fields = ('scene_id', 'provider', 'condition', 'description_model', 'global_schedule_index', 'attempt_id',
              'input_id', 'input_sha256', 'source_label', 'source_plan_identity', 'source_entry_hash',
              'source_gt_descriptor_hash', 'gt_grid_sha256', 'original_attempt_id', 'source_attempt_id', 'selection_reason')
    return {key: row.get(key) for key in fields}


def analyze_factors(slots, *, audit_gate, specification_sha256):
    """Exactly36 cells; B-A is paired on scene within each fixed Builder."""
    require_gate(audit_gate, specification_sha256)
    fixed = audit_gate['input_kind'] == 'verified_sources'
    rows = original.validate_rows(slots, fixed_design=fixed)
    if fixed and Counter(r['status'] for r in rows) != {'success': 934, 'failed': 25, 'operational_unmeasurable': 1}:
        raise ValueError('frozen960 status inventory changed')
    scenes = sorted({r['scene_id'] for r in rows})
    if not scenes or len(scenes) > 60:
        raise ValueError('one to60 independent buildings required')
    selected = [r for r in rows if r['condition'] in CONDITIONS]
    index = {(r['scene_id'], r['provider'], r['condition'], r['description_model']): r for r in selected}
    expected = {(scene, builder, condition, provider) for scene in scenes for builder in BUILDERS
                for condition in CONDITIONS for provider in PROVIDERS}
    if len(index) != len(selected) or set(index) != expected:
        raise ValueError('complete O/J description-provider design required; missing outcomes must be explicit rows')
    records, pairs, exclusions = [], [], []
    for builder in BUILDERS:
        for condition in CONDITIONS:
            for provider_a, provider_b in PAIRS:
                for metric in METRICS:
                    cell_id = ':'.join((builder, condition, provider_b+'-minus-'+provider_a, metric))
                    cell_pairs, cell_exclusions = [], []
                    for scene in scenes:
                        a = index[scene, builder, condition, provider_a]
                        b = index[scene, builder, condition, provider_b]
                        va, vb = original.legacy._metric(a, metric), original.legacy._metric(b, metric)
                        common = {'cell_id': cell_id, 'scene_id': scene, 'a_source': _evidence(a), 'b_source': _evidence(b)}
                        if va is not None and vb is not None:
                            cell_pairs.append({**common, 'a': va, 'b': vb, 'delta': vb-va})
                        else:
                            sides = ['a' if va is None else '', 'b' if vb is None else '']
                            reason = '_and_'.join(side for side in sides if side)+'_unmeasured'
                            cell_exclusions.append({**common, 'reason': reason, 'a_status': a['status'], 'b_status': b['status'],
                                'a_failure_kind': a.get('failure_kind'), 'b_failure_kind': b.get('failure_kind'),
                                'a_metric_available': va is not None, 'b_metric_available': vb is not None})
                    estimates = mean_interval([r['delta'] for r in cell_pairs])
                    records.append({'id': cell_id, 'analysis_label': LABEL, 'builder': builder, 'condition': condition,
                        'description_provider_a': provider_a, 'description_provider_b': provider_b, 'metric': metric,
                        'difference_direction': provider_b+' minus '+provider_a,
                        'metric_favorable_direction': 'lower' if metric == 'aligned_dimension_mae' else 'higher',
                        'planned_buildings': len(scenes), 'common_valid_pairs': len(cell_pairs), 'independent_buildings': len(cell_pairs),
                        'excluded_buildings': len(cell_exclusions), 'exclusion_counts': dict(Counter(r['reason'] for r in cell_exclusions)),
                        'bootstrap_frame': 'common-valid buildings only, sorted by scene_id', 'bootstrap_frame_buildings': len(cell_pairs),
                        'estimator': 'equal-weight mean of one paired B-A difference per building',
                        'p_values_computed': False, **estimates})
                    pairs.extend(cell_pairs); exclusions.extend(cell_exclusions)
    assert len(records) == 36
    return {'protocol_id': PROTOCOL_ID, 'analysis_label': LABEL, 'input_kind': audit_gate['input_kind'],
        'specification_sha256': specification_sha256, 'source_slots_digest': digest(slots), 'records': records,
        'pairs': pairs, 'exclusions': exclusions, 'planned_cells': 36,
        'inference': {'seed': SEED, 'generator': 'NumPy PCG64', 'bootstrap_replicates': REPLICATES,
                      'minimum_valid_replicates': MIN_VALID_REPLICATES, 'interval': '95% percentile, linear quantiles',
                      'independent_unit': 'building', 'p_values': False, 'holm_family_changes': False},
        'limitations': ['Provider differences include image, model-version and decoding-setting confounds.',
                       'O/J are separate sampled Builder calls, not a formatting-only causal intervention.',
                       'Valid-pair estimates condition on successful reconstruction for both descriptions.',
                       'Two Builders repeat the same descriptions and are never combined as120 independent buildings.'],
        'goal007_comparisons_modified_or_reexecuted': False, 'research_api_requests_sent': 0, 'api_key_queries': 0}
