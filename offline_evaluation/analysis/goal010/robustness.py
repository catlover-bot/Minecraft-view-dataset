"""Fixed post-hoc proxy sensitivities; no learned quality score or p-values."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np

from ..goal008 import statistics as frozen
from . import factors

PROTOCOL_ID = 'goal010_proxy_robustness_v1'
WEIGHTS = {'strict_material_f1': .30, 'coarse_material_f1': .25, 'dimension_accuracy': .25, 'completeness': .20}
ABLATIONS = tuple('without_'+feature for feature in WEIGHTS)
FEATURES = (*frozen.FEATURES, *ABLATIONS)
COVARIATES = ('description_chars', 'human_text_chars', 'gt_occupied_count', 'gt_storage_bbox_volume', 'gt_occupancy_density')
METADATA_HASH_FIELDS = ('input_id', *frozen.HASH_FIELDS)


def _metadata_join(joined, metadata_rows, *, fixed_design):
    index = {}
    for row in metadata_rows:
        key = row.get('scene_id'), row.get('description_provider')
        if key in index:
            raise ValueError('duplicate metadata description')
        index[key] = row
    if set(index) != {(r['scene_id'], r['description_provider']) for r in joined}:
        raise ValueError('exact metadata-to-proxy cohort required')
    buildings = {}
    result = deepcopy(joined)
    for row in result:
        metadata = index[row['scene_id'], row['description_provider']]
        if any(metadata.get(field) != row[field] for field in METADATA_HASH_FIELDS):
            raise ValueError('metadata input/source/GT identity mismatch')
        for name in COVARIATES[:-1]:
            value = metadata.get(name)
            if type(value) is not int or value < (0 if name == 'human_text_chars' else 1):
                raise ValueError('nonnegative typed length or positive GT size required: '+name)
        density = metadata.get('gt_occupancy_density')
        if (type(density) not in (int, float) or not math.isfinite(density) or not 0 < density <= 1
                or not math.isclose(density, metadata['gt_occupied_count']/metadata['gt_storage_bbox_volume'], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('GT density must equal occupied/storage volume')
        stratum = metadata.get('size_rank_stratum')
        if type(stratum) is not int or stratum not in range(5):
            raise ValueError('original cohort size-rank stratum0..4 required')
        scene_values = tuple(metadata[k] for k in ('gt_occupied_count', 'gt_storage_bbox_volume', 'gt_occupancy_density', 'size_rank_stratum'))
        if row['scene_id'] in buildings and buildings[row['scene_id']] != scene_values:
            raise ValueError('building GT covariates disagree between description providers')
        buildings[row['scene_id']] = scene_values
        row['metadata'] = {name: metadata[name] for name in (*COVARIATES, 'size_rank_stratum')}
        # Arithmetic component removal only: preserve every stored original score.
        for removed in WEIGHTS:
            kept = {key: weight for key, weight in WEIGHTS.items() if key != removed}
            row['scores']['without_'+removed] = (sum(row['scores'][key]*weight for key, weight in kept.items()) / sum(kept.values())
                if all(row['scores'][key] is not None for key in kept) else None)
        row['scores'].update({name: metadata[name] for name in COVARIATES})
    if fixed_design and Counter(values[-1] for values in buildings.values()) != {i: 12 for i in range(5)}:
        raise ValueError('original60-building size-stratum allocation changed')
    return result


def _tertiles(values):
    """Type7 planned-cohort cutpoints; boundary ties stay in the lower bin."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('nonempty planned covariate frame required')
    cutpoints = np.quantile(values, [1/3, 2/3], method='linear').tolist()
    return {'cutpoints': cutpoints, 'quantile_method': 'linear/type7', 'boundary_rule': 'low <= q1; q1 < middle <= q2; high > q2',
            'planned_values': len(values), 'unique_values': len(set(values.tolist())),
            'boundary_tie_counts': [int(np.sum(values == value)) for value in cutpoints],
            'duplicate_cutpoints': cutpoints[0] == cutpoints[1]}


def _scopes(rows):
    scopes = [('all', rows, {'kind': 'all'})]
    for provider in frozen.PROVIDERS:
        scopes.append(('description_provider:'+provider, [r for r in rows if r['description_provider'] == provider],
                       {'kind': 'description_provider', 'description_provider': provider}))
    for stratum in range(5):
        scopes.append(('size_rank_stratum:'+str(stratum), [r for r in rows if r['metadata']['size_rank_stratum'] == stratum],
                       {'kind': 'original_size_rank_stratum', 'size_rank_stratum': stratum}))
    diagnostic_scopes = list(scopes)
    bins = {}
    for covariate in ('description_chars', 'gt_storage_bbox_volume', 'gt_occupancy_density'):
        planned = ([r['metadata'][covariate] for r in rows] if covariate == 'description_chars' else
                   list({r['scene_id']: r['metadata'][covariate] for r in rows}.values()))
        definition = _tertiles(planned)
        definition['sampling_frame'] = 'planned descriptions' if covariate == 'description_chars' else 'planned unique buildings'
        bins[covariate] = definition
        for value in range(3):
            selected = [r for r in rows if int(np.searchsorted(definition['cutpoints'], r['metadata'][covariate], side='left')) == value]
            scopes.append((covariate+':tertile:'+str(value), selected,
                           {'kind': 'planned_covariate_tertile', 'covariate': covariate, 'tertile': value}))
    assert len(scopes) == 18 and len(diagnostic_scopes) == 9
    return scopes, diagnostic_scopes, bins


def _describe(rows):
    records = []
    for variable in COVARIATES:
        for provider in (None, *frozen.PROVIDERS):
            selected = [r for r in rows if provider is None or r['description_provider'] == provider]
            by_building = {r['scene_id']: r['metadata'][variable] for r in selected}
            values = ([r['metadata'][variable] for r in selected] if variable.endswith('_chars') else list(by_building.values()))
            records.append({'covariate': variable, 'description_provider': provider,
                'observation_unit': 'description' if variable.endswith('_chars') else 'building',
                'observations': len(values), 'buildings': len(by_building), 'minimum': min(values) if values else None,
                'median': float(np.median(values)) if values else None, 'maximum': max(values) if values else None,
                'unique_values': len(set(values))})
    return records


def analyze_robustness(proxy_rows, slots, metadata_rows, *, audit_gate, specification_sha256):
    factors.require_gate(audit_gate, specification_sha256)
    fixed = audit_gate['input_kind'] == 'verified_sources'
    joined = frozen.join_descriptions(proxy_rows, slots, fixed_design=fixed)
    rows = _metadata_join(joined, metadata_rows, fixed_design=fixed)
    scopes, diagnostic_scopes, bins = _scopes(rows)
    frame_cache, records, scope_counts = {}, [], []

    def analyze(scope, feature, target, policy, method, group):
        scope_id, selected, scope_detail = scope
        scenes = tuple(sorted({r['scene_id'] for r in selected}))
        if scenes not in frame_cache:
            frame_cache[scenes] = factors.draw_counts(len(scenes))
        scene_index = {scene: index for index, scene in enumerate(scenes)}
        # Frozen arithmetic helpers are reused without touching the old protocol,
        # old draws, old data or its360 result records. New draws use20261017.
        result = frozen._association(selected, feature, target, policy, method, None, scene_index, frame_cache[scenes])
        result.update(id=':'.join((group, scope_id, policy, target, feature, method)),
            analysis_label=factors.LABEL, analysis_group=group, scope_id=scope_id, scope=scope_detail,
            description_provider=scope_detail.get('description_provider'),
            feature_origin=('frozen_component_weight_removal_sensitivity' if feature in ABLATIONS else
                            'observed_covariate' if feature in COVARIATES else 'unchanged_frozen_goal008_feature'),
            bootstrap_frame='all planned buildings represented in this scope; duplicate draws are separate cluster copies',
            bootstrap_frame_buildings=len(scenes), bootstrap_frame_sha256=factors.digest(scenes),
            old_goal008_result_replaced=False)
        records.append(result)

    for scope in scopes:
        scope_id, selected, detail = scope
        scope_counts.append({'scope_id': scope_id, **detail, 'planned_descriptions': len(selected),
                             'planned_buildings': len({r['scene_id'] for r in selected})})
        for feature in FEATURES:
            for target in frozen.TARGETS:
                for policy in frozen.POLICIES:
                    analyze(scope, feature, target, policy, 'pooled_spearman', 'proxy_stratified')
    for feature in FEATURES:
        for target in frozen.TARGETS:
            for policy in frozen.POLICIES:
                for method in ('within_building_centered_pearson', 'within_building_midrank_association'):
                    analyze(scopes[0], feature, target, policy, method, 'proxy_within_building')
    for scope in diagnostic_scopes:
        for feature in COVARIATES:
            for target in frozen.TARGETS:
                for policy in frozen.POLICIES:
                    analyze(scope, feature, target, policy, 'pooled_spearman', 'covariate_outcome')
    assert len(records) == len({r['id'] for r in records}) == 1350
    return {'protocol_id': PROTOCOL_ID, 'analysis_label': factors.LABEL, 'input_kind': audit_gate['input_kind'],
        'specification_sha256': specification_sha256, 'records': records, 'planned_records': 1350,
        'record_group_counts': dict(Counter(r['analysis_group'] for r in records)),
        'status_counts': dict(Counter(r['status'] for r in records)), 'scope_counts': scope_counts, 'tertile_definitions': bins,
        'covariate_descriptions': _describe(rows), 'metadata_digest': factors.digest(metadata_rows),
        'source_binding': {'descriptions': len(rows), 'buildings': len({r['scene_id'] for r in rows}),
                           'source_joined_digest': factors.digest(joined), 'builder_observations': len(rows)*2},
        'inference': {'seed': factors.SEED, 'generator': 'NumPy PCG64', 'bootstrap_replicates': factors.REPLICATES,
            'minimum_valid_replicates': factors.MIN_VALID_REPLICATES, 'interval': '95% percentile, linear quantiles',
            'independent_unit': 'building', 'p_values': False, 'holm_family_changes': False,
            'weighting': 'equal eligible description pairs; whole-building cluster uncertainty',
            'strata_membership': 'fixed from planned cohort before outcomes, never re-binned per resample'},
        'component_removal_definition': {'weights': WEIGHTS, 'rule': 'drop one frozen component then renormalize remaining fixed weights; any remaining null stays null',
            'interpretation': 'arithmetic sensitivity, not a new validated quality score or independent causal contribution'},
        'limitations': ['Covariate strata condition descriptively and do not isolate causal effects or independent component contributions.',
            'Original occupied-count rank strata come from the100-building population; each actual evaluation stratum contains12 buildings.',
            'Ties remain in the same tertile; empty or constant strata and unstable intervals stay in the full result grid.',
            'Two Builders repeat each description; independent N is at most60 buildings, not120 or360.',
            'Shared GT, selected valid outcomes and multiple exploratory analyses limit interpretation.',
            'No fitted quality model, regression adjustment, grouped CV, p-values or new Holm family is used.'],
        'historical_rq2_status': 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES',
        'goal007_comparisons_modified_or_reexecuted': False, 'goal008_frozen_results_modified': False,
        'research_api_requests_sent': 0, 'api_key_queries': 0}
