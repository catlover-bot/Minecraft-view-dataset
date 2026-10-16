"""Synthetic-only publication and source-receipt regression tests."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path

import pytest

from offline_evaluation.analysis.goal008 import publication as pub


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pub._json(value) if not isinstance(value, str) else value)


def _csv(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, list(rows[0]), lineterminator='\n'); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue()


def _products(root):
    return {str(p.relative_to(root)): {'sha256': pub.sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file()}


def synthetic_results():
    groups, denominators = [], []
    metrics = set(pub.MAIN_METRICS) | {'absolute_occupancy_iou', 'aligned_material_set_f1',
                                    'aligned_add_count', 'aligned_delete_count', 'aligned_replace_count'}
    for builder in pub.BUILDERS:
        for condition in pub.CONDITIONS:
            planned = 6 if condition in 'OJ' else 2
            failed = int(condition == 'E' or (condition == 'O' and builder == 'openai'))
            valid = planned - failed
            den = {'provider': builder, 'condition': condition, 'description_model': None,
                   'group_level': 'builder_condition', 'planned_slots': planned,
                   'operational_unmeasurable_slots': 0,
                   'failure_kind_counts': {'truncation' if condition == 'E' else 'validation_failure': failed}}
            denominators.append(den)
            for metric in sorted(metrics):
                groups.append({**den, 'metric': metric, 'equal_building_mean': .2 if 'iou' in metric else 1.2,
                    'valid_rows': valid, 'valid_buildings': 1 if condition == 'E' else 2,
                    'failure0_eligible_rows': planned, 'failure0_equal_building_mean': .1})
    comparisons = []
    pairs = [('primary_builder_valid_pair', 'openai', 'anthropic', c, c,
              'anthropic-minus-openai:' + c) for c in pub.CONDITIONS]
    pairs += [('secondary_lossless_J_minus_O', b, b, 'O', 'J', 'J-minus-O:' + b) for b in pub.BUILDERS]
    pairs += [('diagnostic_oracle_minus_original', b, b, 'O', c, c + '-minus-O:' + b)
              for b in pub.BUILDERS for c in ('P', 'E')]
    for family, a, b, ac, bc, id in pairs:
        for metric in ('aligned_occupancy_iou', 'aligned_dimension_mae'):
            comparisons.append({'id': id + ':' + metric, 'family_id': family, 'family_size': pub.FAMILIES[family],
                'a_provider': a, 'b_provider': b, 'a_condition': ac, 'b_condition': bc, 'metric': metric,
                'mean_delta': .0123456789, 'ci95': [-.1, .23456789], 'p_two_sided': .0000999900009999,
                'p_holm': .45, 'common_valid_pair_count': 5, 'buildings': 2})
    return {'input_kind': 'fixture', 'comparisons': comparisons, 'per_group': groups,
            'denominators': {'groups': denominators, 'overall': {'planned_slots': 32, 'independent_buildings': 2,
                'valid_only_aligned_iou_slots': 29, 'model_output_failure_slots': 3,
                'operational_unmeasurable_slots': 0, 'completed_request_outcome_slots': 32}}}


def synthetic_associations():
    joined = []
    for scene in range(2):
        for index, provider in enumerate(pub.PROVIDERS):
            missing = scene == 0 and index == 0
            value = .1 + .1 * index + .05 * scene
            joined.append({'scene_id': 'scene_' + str(scene), 'description_provider': provider,
                'input_id': f'input_{scene}_{provider}', 'input_sha256': 'a' * 64, 'source_sha256': 'b' * 64,
                'gt_bbox_sha256': 'c' * 64, 'gt_voxels_sha256': 'd' * 64,
                'scores': {name: value for name in pub.FEATURES},
                'builders': {b: {'valid_only': None if missing and b == 'openai' else value,
                    'bounded_iou_failure0': 0 if missing and b == 'openai' else value} for b in pub.BUILDERS}})
    records = []
    for scope in (None, *pub.PROVIDERS):
        for policy in pub.POLICIES:
            for target in pub.TARGETS:
                for feature in pub.FEATURES:
                    for association in pub.ASSOCIATIONS:
                        rows = [r for r in joined if scope is None or r['description_provider'] == scope]
                        complete = [r for r in rows if all(r['builders'][b][policy] is not None
                                    for b in (pub.BUILDERS if target == 'both_builders_mean' else (target,)))]
                        nonestimable = scope is not None and association != 'pooled_spearman'
                        count = 0 if nonestimable else len(complete)
                        records.append({'id': ':'.join((scope or 'all', policy, target, feature, association)),
                            'description_provider': scope, 'outcome_policy': policy, 'outcome_target': target,
                            'feature': feature, 'association': association, 'p_values_computed': False,
                            'coefficient': None if nonestimable else .25, 'ci95': None if nonestimable else [-.2, .6],
                            'counts': {'analysis_pairs': count, 'analysis_buildings': 0 if nonestimable else 2,
                                       'missing_outcome': len(rows)-len(complete)},
                            'bootstrap_valid': 0 if nonestimable else 2000, 'bootstrap_planned': 2000,
                            'status': 'NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING' if nonestimable else 'ESTIMABLE'})
    return {'input_kind': 'fixture', 'conventional_rq2_status': pub.RQ2_STATUS, 'joined_descriptions': joined,
            'associations': records, 'inference': {'p_values': False, 'holm_family_changes': False},
            'status_counts': {'ESTIMABLE': 180, 'NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING': 180}}


@pytest.fixture
def package(tmp_path):
    old, phase, audit = (tmp_path / name for name in ('old', 'phase', 'audit'))
    r, a = synthetic_results(), synthetic_associations()
    _write(old / 'results/results.json', r)
    _write(old / 'data/final_slots_960.json', [])
    _write(old / 'results/comparisons_20.csv', _csv(r['comparisons']))
    _write(old / 'results/rq2_matched_scores_audit.json', {'status': pub.RQ2_STATUS})
    _write(old / 'MANIFEST.json', {'products': _products(old)})
    _write(audit / 'representation_semantics.json', {'O_J_typed_tree_equal_count': 6})
    _write(audit / 'sampling_audit.json', {'status': 'PASS', 'O_attempts': 7, 'J_attempts': 6})
    gate = {'status': 'PASS', 'goal007_manifest_sha256': pub.sha256(old / 'MANIFEST.json'),
            'files': {str(audit / k): v for k, v in _products(audit).items()},
            'fixed_scope_attempts': 33, 'historical_calls': 63, 'budget_accounted_usd': '0.00000000'}
    _write(audit / 'source_gate.json', gate)
    _write(phase / 'proxy_scores_180.json', [{k:v for k,v in row.items() if k != 'builders'} for row in a['joined_descriptions']])
    _write(phase / 'exploratory_associations.json', a)
    _write(phase / 'exploratory_associations_360.csv', _csv(a['associations']))
    _write(phase / 'PHASE1_MANIFEST.json', {'status': 'PHASE0_PHASE1_PASS', 'created_code_commit': 'a' * 40,
        'source_gate_sha256': pub.sha256(audit / 'source_gate.json'), 'goal007_manifest_sha256': pub.sha256(old / 'MANIFEST.json'),
        'goal007_slots_sha256': pub.sha256(old / 'data/final_slots_960.json'),
        'goal007_comparisons_sha256': pub.sha256(old / 'results/comparisons_20.csv'), 'products': _products(phase)})
    return old, phase, audit


def test_compact_tables_keep_all_exact_statistics(package):
    data = pub.load_inputs(*package)
    docs, numeric = pub.build_tables(data['results'], data['associations'])
    assert len(numeric['main_rows']) == len(numeric['failure_rows']) == 8
    assert len(numeric['frozen_comparisons']) == 20
    assert len(numeric['all_exploratory_associations']) == 360
    assert docs['stats_appendix.tex'].count(r'\begin{table*}') == 3
    assert docs['exploratory_appendix.tex'].count(r'\begin{table*}') == 18
    assert docs['ablations.tex'].count('0.0123') == 12
    assert docs['stats_appendix.tex'].count('0.0123') == 20
    assert 'NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING' not in docs['exploratory_appendix.tex']
    assert '1/cluster' in docs['exploratory_appendix.tex']
    assert any(r['status']=='NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING' for r in numeric['all_exploratory_associations'])
    assert numeric['frozen_comparisons'][0]['mean_delta'] == .0123456789
    assert '9.999e-05' in docs['stats_appendix.tex']
    assert 'p=0' not in docs['stats_appendix.tex']


@pytest.mark.parametrize('value', [0, -1, 1.01, float('nan'), float('inf'), True])
def test_zero_or_invalid_p_never_displayed(value):
    with pytest.raises(ValueError): pub.p_value(value)


def test_tex_escapes_untrusted_text():
    assert pub.tex('a_b&c%#${}\\') == r'a\_b\&c\%\#\$\{\}\textbackslash{}'


def test_complete_pair_scatter_does_not_impute_or_pool_builders():
    rows = synthetic_associations()['joined_descriptions']
    before = deepcopy(rows)
    points = pub.scatter_rows(rows)
    assert len(points) == 5
    assert len({p['input_id'] for p in points}) == 5
    assert not any(p['input_id'] == 'input_0_openai' for p in points)
    for scene in ('scene_0', 'scene_1'):
        assert abs(sum(r['centered_composite'] for r in points if r['scene_id']==scene)) < 1e-12
    assert rows == before
    singleton = pub.scatter_rows(rows[1:2])
    assert singleton[0]['centered_composite'] is None


def test_changed_phase1_product_stops_before_output(package, tmp_path):
    old, phase, audit = package
    with (phase / 'proxy_scores_180.json').open('a') as f: f.write(' ')
    target = tmp_path / 'output'
    with pytest.raises(ValueError, match='hash or size mismatch'):
        pub.render_publication(old, phase, audit, target)
    assert not target.exists()


def test_nonpassing_phase1_stops_before_output(package, tmp_path):
    path = package[1] / 'PHASE1_MANIFEST.json'
    receipt = pub.read_json(path); receipt['status'] = 'RUNNING'; _write(path, receipt)
    with pytest.raises(ValueError, match='Phase0 and Phase1 PASS'):
        pub.render_publication(*package, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_source_join_tamper_is_rejected(package):
    data = pub.load_inputs(*package)
    data['scores'][0]['source_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='source binding changed'): pub.validate_inputs(data)


def test_omitted_nonsignificant_record_is_rejected(package):
    data = pub.load_inputs(*package)
    data['associations']['associations'].pop()
    with pytest.raises(ValueError, match='all360'): pub.validate_inputs(data)


def test_historical_rq2_cannot_be_reclassified(package):
    data = pub.load_inputs(*package); data['rq2']['status'] = 'TESTED'
    with pytest.raises(ValueError, match='RQ2 status'): pub.validate_inputs(data)


def test_only_manifest_bound_rq2_is_read_and_watched(package):
    _write(package[0] / 'rq2_matched_scores_audit.json', {'status': 'UNBOUND_COPY'})
    data = pub.load_inputs(*package)
    assert data['rq2']['status'] == pub.RQ2_STATUS
    assert str((package[0] / 'results/rq2_matched_scores_audit.json').absolute()) in data['input_files']
    with (package[0] / 'results/rq2_matched_scores_audit.json').open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError, match='hash or size mismatch'): pub.load_inputs(*package)


def test_production_missing942_and_counts_are_gated(package):
    data = pub.load_inputs(*package)
    data['results']['input_kind'] = data['associations']['input_kind'] = 'verified_sources'
    with pytest.raises(ValueError, match='frozen960'): pub.validate_inputs(data)


def test_csv_json_numeric_disagreement_is_rejected(package):
    data = pub.load_inputs(*package)
    data['results']['comparisons'][0]['p_holm'] = .5
    with pytest.raises(ValueError, match='CSV/JSON statistics differ'): pub.validate_inputs(data)


def test_distinct_main_denominators_are_not_hidden(package):
    data = pub.load_inputs(*package)
    next(r for r in data['results']['per_group'] if r['metric']=='aligned_dimension_mae')['valid_rows'] -= 1
    with pytest.raises(ValueError, match='distinct denominators'): pub.build_tables(data['results'], data['associations'])


def test_frozen_error_metric_rows_retain_denominators(package):
    data = pub.load_inputs(*package)
    docs, numeric = pub.build_tables(data['results'], data['associations'])
    assert len(numeric['error_metric_rows']) == 16
    assert {r['metric'] for r in numeric['error_metric_rows']} == set(numeric['error_metrics'])
    assert all('valid_rows' in r and 'valid_buildings' in r for r in numeric['error_metric_rows'])
    assert 'Slots / buildings' in docs['error_structure.tex']


def test_optional_examples_are_bound_copied_and_cited(package, tmp_path, monkeypatch):
    monkeypatch.setattr(pub.shutil, 'which', lambda executable: None)
    monkeypatch.setattr(pub, 'plot_figures', lambda *a, **kw: {'scatter_points': [], 'figures': []})
    examples = tmp_path / 'examples'
    for name in ('example_01.png', 'example_02.png', 'example_03.png', 'examples_figure.tex'):
        _write(examples / name, 'synthetic fixture ' + name)
    manifest = {'status': 'RENDERED', 'input_kind': 'fixture', 'examples': ['a', 'b', 'c'],
                'files': [{'path': k, **v} for k, v in _products(examples).items()]}
    _write(examples / 'example_manifest.json', manifest)
    out = tmp_path / 'with_examples'
    pub.render_publication(*package, out, examples_manifest=examples / 'example_manifest.json')
    for name in ('example_01.png', 'example_02.png', 'example_03.png', 'examples_figure.tex'):
        assert (out / name).read_bytes() == (examples / name).read_bytes()
    assert r'\input{examples_figure.tex}' in (out / 'figure_captions.tex').read_text()
    assert pub.read_json(out / 'RUN.json')['examples_included'] == ['a', 'b', 'c']
    with (examples / 'example_01.png').open('a') as stream: stream.write('changed')
    with pytest.raises(ValueError, match='hash or size mismatch'):
        pub.render_publication(*package, tmp_path / 'bad_examples', examples_manifest=examples / 'example_manifest.json')
    assert not (tmp_path / 'bad_examples').exists()


def test_path_traversal_manifest_product_is_rejected(package, tmp_path):
    path = package[1] / 'PHASE1_MANIFEST.json'
    receipt = pub.read_json(path)
    receipt['products']['../not-a-product'] = {'sha256': 'a' * 64, 'bytes': 1}
    _write(path, receipt)
    with pytest.raises(ValueError, match='unsafe product path'):
        pub.render_publication(*package, tmp_path / 'unsafe')


def test_real_completion_figures_receipts_and_sources_unchanged(package, tmp_path, monkeypatch):
    monkeypatch.setattr(pub.shutil, 'which', lambda executable: None)
    before = {str(p): pub.sha256(p) for root in package for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'publication'
    receipt = pub.render_publication(*package, out)
    assert receipt['status'] == 'PUBLICATION_RENDERED'
    assert receipt['run_sha256'] == pub.sha256(out / 'RUN.json')
    assert (out / 'RUN.sha256').read_text().startswith(receipt['run_sha256'])
    run = pub.read_json(out / 'RUN.json')
    assert run['latex_validation']['status'] == 'NOT_TESTED_COMPILER_UNAVAILABLE'
    assert run['new_inference_runs'] == run['research_api_requests_sent'] == run['source_mutations'] == 0
    assert before == {p: pub.sha256(p) for p in before}
    for name, descriptor in run['products'].items():
        assert pub.sha256(out / name) == descriptor['sha256']
    assert len(pub.read_json(out / 'proxy_scatter_points.json')) == 5
    for name in ('figure_main_errors', 'figure_proxy_scatter'):
        assert (out / (name + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        svg = (out / (name + '.svg')).read_text()
        assert '<svg' in svg and 'SYNTHETIC FIXTURE' in svg
    assert pub.PROXY_LABEL in (out / 'figure_proxy_scatter.svg').read_text()
    assert pub.RQ2_STATUS in (out / 'Results.md').read_text()
    assert 'post-hoc' in (out / 'RQ_REFRAMING.md').read_text()
    assert 'no IoU=0 is imputed' in (out / 'Results.md').read_text()
    assert (out / 'exploratory_associations_360.csv').read_bytes() == (package[1] / 'exploratory_associations_360.csv').read_bytes()
    with pytest.raises(ValueError, match='new publication directory'):
        pub.render_publication(*package, out)
