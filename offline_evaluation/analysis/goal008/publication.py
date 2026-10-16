"""Publication from frozen Goal007/008 results, without inference or source writes.

Only matplotlib and NumPy are imported lazily for plots.  No scorer, evaluator,
statistics module, API client, credential helper or network library is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

VERSION = 'goal008_publication_v1'
PROXY_LABEL = 'Exploratory CPU attribute proxy—not conventional description quality'
RQ2_STATUS = 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES'
BUILDERS = ('openai', 'anthropic')
PROVIDERS = (*BUILDERS, 'gemini')
CONDITIONS = ('O', 'J', 'P', 'E')
FEATURES = ('strict_material_f1', 'coarse_material_f1', 'dimension_accuracy', 'completeness', 'composite')
ASSOCIATIONS = ('pooled_spearman', 'within_building_centered_pearson', 'within_building_midrank_association')
POLICIES = ('valid_only', 'bounded_iou_failure0')
TARGETS = (*BUILDERS, 'both_builders_mean')
FAMILIES = {'primary_builder_valid_pair': 8, 'secondary_lossless_J_minus_O': 4,
            'diagnostic_oracle_minus_original': 8}
NAMES = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'gemini': 'Gemini',
         'both_builders_mean': 'Both-Builder mean'}
FEATURE_NAMES = {'strict_material_f1': 'Strict material F1', 'coarse_material_f1': 'Coarse material F1',
                 'dimension_accuracy': 'Dimension accuracy', 'completeness': 'Field completeness', 'composite': 'Composite'}
ASSOCIATION_NAMES = {'pooled_spearman': 'Pooled Spearman', 'within_building_centered_pearson': 'Within-building centered',
                     'within_building_midrank_association': 'Within-building rank'}
STATUS_LABELS = {'ESTIMABLE': 'OK', 'ESTIMATE_WITHOUT_CLUSTER_CI': 'CI?',
                 'CONSTANT_PROXY': 'C-X', 'CONSTANT_OUTCOME': 'C-Y', 'CONSTANT_BOTH': 'C-XY',
                 'NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING': '1/cluster',
                 'INSUFFICIENT_PAIRS': 'N<2', 'NUMERICALLY_UNDEFINED': 'Numeric?'}
COMPACT_FEATURES = {'strict_material_f1': 'Strict mat.', 'coarse_material_f1': 'Coarse mat.',
                    'dimension_accuracy': 'Dimensions', 'completeness': 'Completeness', 'composite': 'Composite'}
MAIN_METRICS = ('aligned_occupancy_iou', 'aligned_material_aware_iou',
                'aligned_dimension_mae', 'aligned_total_normalized_repair_cost')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def _new(path, content):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(content)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('displayed values must be finite numbers or explicit null')
    return value


def number(value, digits=4):
    return 'NA' if value is None else f'{_finite(value):.{digits}f}'


def p_value(value):
    if value is None:
        return 'NA'
    if not 0 < _finite(value) <= 1:
        raise ValueError('positive finite observed p-value required; never display p=0')
    return f'{value:.6g}'


def interval(value):
    return 'NA' if value is None else '[' + ', '.join(number(v) for v in value) + ']'


def tex(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
               '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '—': '---', '−': '-'}
    return ''.join(mapping.get(c, c) for c in str(value))


def table(headers, rows, caption, label):
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\caption{' + tex(caption) + '}', r'\label{' + label + '}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
             ' & '.join(tex(v) for v in headers) + r' \\', r'\midrule']
    lines += [' & '.join(tex(v) for v in row) + r' \\' for row in rows]
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])


def _regular(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError('regular nonsymlink input required: ' + str(path))
    return path


def _product(root, name, descriptor):
    name = Path(name)
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('unsafe product path')
    path = _regular(Path(root) / name)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('product escaped source directory')
    if sha256(path) != descriptor['sha256'] or path.stat().st_size != descriptor['bytes']:
        raise ValueError('source product hash or size mismatch: ' + str(name))
    return path


def load_inputs(goal007_directory, phase1_directory, audit_directory):
    """Check frozen result receipts before creating a publication directory."""
    old, phase, audit = map(Path, (goal007_directory, phase1_directory, audit_directory))
    manifest_path, phase_path, gate_path = old / 'MANIFEST.json', phase / 'PHASE1_MANIFEST.json', audit / 'source_gate.json'
    manifest, receipt, gate = map(read_json, (manifest_path, phase_path, gate_path))
    if receipt.get('status') != 'PHASE0_PHASE1_PASS' or gate.get('status') != 'PASS':
        raise ValueError('completed Phase0 and Phase1 PASS required before publication')
    if receipt['source_gate_sha256'] != sha256(gate_path):
        raise ValueError('Phase1 source gate binding mismatch')
    if sha256(manifest_path) != receipt['goal007_manifest_sha256'] or gate['goal007_manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Goal007 manifest binding mismatch')
    files = [manifest_path, phase_path, gate_path]
    for root, records in ((old, manifest['products']), (phase, receipt['products'])):
        for name, descriptor in records.items():
            files.append(_product(root, name, descriptor))
    for name in ('representation_semantics.json', 'sampling_audit.json'):
        path = _regular(audit / name)
        descriptors = [v for k, v in gate['files'].items() if Path(k).name == name]
        if len(descriptors) != 1 or descriptors[0]['sha256'] != sha256(path) or descriptors[0]['bytes'] != path.stat().st_size:
            raise ValueError('Phase0 semantics/sampling binding mismatch')
        files.append(path)
    for name, key in (('data/final_slots_960.json', 'goal007_slots_sha256'), ('results/comparisons_20.csv', 'goal007_comparisons_sha256')):
        if sha256(old / name) != receipt[key]:
            raise ValueError('frozen Goal007 result binding mismatch')
    result = {'results': read_json(old / 'results/results.json'), 'slots': read_json(old / 'data/final_slots_960.json'),
              'rq2': read_json(old / 'results/rq2_matched_scores_audit.json'), 'phase1': receipt, 'gate': gate,
              'scores': read_json(phase / 'proxy_scores_180.json'),
              'associations': read_json(phase / 'exploratory_associations.json'),
              'semantics': read_json(audit / 'representation_semantics.json'), 'sampling': read_json(audit / 'sampling_audit.json'),
              'input_files': {str(_regular(p)): sha256(p) for p in sorted(set(files))},
              'comparisons_csv_path': old / 'results/comparisons_20.csv',
              'exploratory_csv_path': phase / 'exploratory_associations_360.csv'}
    validate_inputs(result)
    return result


def validate_inputs(data):
    results, associations = data['results'], data['associations']
    kind = results.get('input_kind')
    if kind not in ('fixture', 'verified_sources') or associations.get('input_kind') != kind:
        raise ValueError('matching explicit input kind required')
    if data['rq2'].get('status') != RQ2_STATUS or associations.get('conventional_rq2_status') != RQ2_STATUS:
        raise ValueError('historical RQ2 status must remain unchanged')
    comparisons = results['comparisons']
    if len({r['id'] for r in comparisons}) != 20 or Counter(r['family_id'] for r in comparisons) != FAMILIES:
        raise ValueError('exact original twenty contrasts and 8/4/8 families required')
    for row in comparisons:
        p_value(row['p_two_sided']); p_value(row['p_holm'])
        _finite(row['mean_delta'])
        if row['buildings'] > 60 or row['family_size'] != FAMILIES[row['family_id']]:
            raise ValueError('invalid independent N or confirmatory family')
    with Path(data['comparisons_csv_path']).open(newline='', encoding='utf-8') as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_index = {r['id']: r for r in csv_rows}
    if len(csv_rows) != 20 or set(csv_index) != {r['id'] for r in comparisons}:
        raise ValueError('Goal007 CSV/JSON contrast IDs differ')
    for row in comparisons:
        other = csv_index[row['id']]
        if any(float(other[k]) != row[k] for k in ('mean_delta', 'p_two_sided', 'p_holm')) or json.loads(other['ci95']) != row['ci95']:
            raise ValueError('Goal007 CSV/JSON statistics differ')
    expected = {(scope, policy, target, feature, association) for scope in (None, *PROVIDERS)
                for policy in POLICIES for target in TARGETS for feature in FEATURES for association in ASSOCIATIONS}
    records = associations['associations']
    actual = [(r['description_provider'], r['outcome_policy'], r['outcome_target'], r['feature'], r['association']) for r in records]
    if len(actual) != 360 or set(actual) != expected or any(r.get('p_values_computed') is not False for r in records):
        raise ValueError('all360 fixed exploratory records without new p-values required')
    if associations['inference'].get('p_values') is not False or associations['inference'].get('holm_family_changes') is not False:
        raise ValueError('exploratory results cannot extend confirmatory families')
    joined = associations['joined_descriptions']
    keys = [(r['scene_id'], r['description_provider']) for r in joined]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicated source-bound description')
    score_index = {(r['scene_id'], r['description_provider']): r for r in data['scores']}
    if set(score_index) != set(keys) or len(score_index) != len(data['scores']):
        raise ValueError('score and joined description cohort mismatch')
    for row in joined:
        score = score_index[(row['scene_id'], row['description_provider'])]
        if any(row[k] != score[k] for k in ('input_id', 'input_sha256', 'source_sha256', 'gt_bbox_sha256', 'gt_voxels_sha256', 'scores')):
            raise ValueError('score to outcome source binding changed')
    if data['semantics'].get('O_J_typed_tree_equal_count') != len(joined) or data['sampling'].get('status') != 'PASS':
        raise ValueError('representation and sampling audit required')
    if kind == 'verified_sources':
        overall = results['denominators']['overall']
        expected_counts = {'planned_slots': 960, 'valid_only_aligned_iou_slots': 934, 'model_output_failure_slots': 25,
                           'operational_unmeasurable_slots': 1, 'completed_request_outcome_slots': 959, 'independent_buildings': 60}
        if any(overall[k] != v for k, v in expected_counts.items()) or len(data['slots']) != 960 or len(joined) != 180:
            raise ValueError('frozen960/934/25/1 cohort changed')
        missing = [r for r in data['slots'] if r['global_schedule_index'] == 942]
        if len(missing) != 1 or missing[0]['metrics'] is not None or missing[0]['status'] != 'operational_unmeasurable':
            raise ValueError('global942 must remain unmeasured, never failure0')
        if data['sampling']['O_attempts'] != 361 or data['sampling']['J_attempts'] != 360:
            raise ValueError('O/J call frequency changed')
        if len(scatter_rows(joined)) != 176:
            raise ValueError('both-valid description count changed')


def _groups(results):
    rows = [r for r in results['per_group'] if r['group_level'] == 'builder_condition']
    index = {(r['provider'], r['condition'], r['metric']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate aggregate metric')
    return index


def _contrast(row):
    if row['a_provider'] != row['b_provider']:
        return 'Anthropic - OpenAI / ' + row['a_condition']
    return row['b_condition'] + ' - ' + row['a_condition'] + ' / ' + NAMES[row['a_provider']]


def build_tables(results, associations):
    """Round only presentation; preserve all exact source records in JSON."""
    groups = _groups(results)
    denominators = {(r['provider'], r['condition']): r for r in results['denominators']['groups']
                    if r['group_level'] == 'builder_condition'}
    main, failures, main_exact, failure_exact = [], [], [], []
    for condition in CONDITIONS:
        for builder in BUILDERS:
            values = [groups[builder, condition, metric] for metric in MAIN_METRICS]
            row, den = values[0], denominators[builder, condition]
            if any((v['valid_rows'], v['valid_buildings']) != (row['valid_rows'], row['valid_buildings']) for v in values):
                raise ValueError('main metrics have distinct denominators; table must show them separately')
            main.append([NAMES[builder], condition, f"{row['valid_rows']}/{row['planned_slots']}", row['valid_buildings'],
                         *[number(v['equal_building_mean']) for v in values]])
            kinds = den['failure_kind_counts']
            failures.append([NAMES[builder], condition, f"{row['valid_rows']}/{row['planned_slots']}",
                kinds.get('truncation', 0), kinds.get('validation_failure', 0), den['operational_unmeasurable_slots'],
                row['failure0_eligible_rows'], number(row['failure0_equal_building_mean'])])
            main_exact.append({'provider': builder, 'condition': condition, 'metrics': {v['metric']: v for v in values}})
            failure_exact.append({'provider': builder, 'condition': condition, 'denominators': den, 'iou': row})
    main_tex = table(['Builder', 'Cond.', 'Valid / planned', 'Buildings', 'IoU', 'Mat. IoU', 'MAE', 'Repair'], main,
        'Strict aligned metrics, with equal weight per building after within-building aggregation. IoU and exact-material-aware voxel IoU: higher is better; dimension MAE (blocks) and normalized repair cost: lower is better. Repair counts operations, not human time. All means condition on valid outputs; n is metric-specific independent buildings (at most 60). O/J use three description providers; P/E have one GT-derived input per building.', 'tab:goal008-main')
    failure_tex = table(['Builder', 'Cond.', 'Valid / planned', 'Trunc.', 'Validation', 'Missing', 'F0 eligible', 'F0 IoU'], failures,
        'Final selected outcomes. F0 is a separate bounded aligned-IoU sensitivity: only identified completed model-output failures receive zero. Operational HTTP 520 at slot 942 remains null and is excluded from F0. F0 means weight buildings equally. Failure counts are slots, not independent buildings. The original HTTP 401 and approved retry belong to the attempt audit, not two selected slots.', 'tab:goal008-failures')
    def contrast_rows(rows, raw=False):
        return [[_contrast(r), 'IoU' if r['metric'] == 'aligned_occupancy_iou' else 'MAE', number(r['mean_delta']),
                 interval(r['ci95']), *([p_value(r['p_two_sided'])] if raw else []), p_value(r['p_holm']),
                 r['common_valid_pair_count'], r['buildings']] for r in rows]
    ablations = [r for r in results['comparisons'] if r['family_id'] != 'primary_builder_valid_pair']
    ablation_tex = table(['Contrast', 'Metric', 'Delta', '95% CI', 'Holm p', 'Pairs', 'Buildings'], contrast_rows(ablations),
        'All 12 frozen J-O and diagnostic P/E-O contrasts. Delta is the named first condition minus O on common valid pairs; positive IoU favors the first condition, positive MAE favors O. Building bootstrap: 2,000 draws; sign-flip: 10,000 draws; Holm within the original families of 4 and 8. Nonsignificance is not equivalence. No exploratory proxy tests enter these families.', 'tab:goal008-ablations')
    family_names = {'primary_builder_valid_pair': 'Builder contrasts (family size 8)', 'secondary_lossless_J_minus_O': 'Lossless JSON reserialization (family size 4)', 'diagnostic_oracle_minus_original': 'GT diagnostic contrasts (family size 8)'}
    appendix = ('\n' + r'\clearpage' + '\n').join(table(['Contrast', 'Metric', 'Delta', '95% CI', 'Raw p', 'Holm p', 'Pairs', 'Buildings'],
        contrast_rows([r for r in results['comparisons'] if r['family_id'] == family], True),
        family_names[family] + '. Unmodified Goal007 paired estimates, building-bootstrap intervals and sign-flip p-values. Direction is B-A; IoU higher and MAE lower are favorable. Pair and independent building denominators differ. Exact values and methods are retained in frozen_comparisons_20.json.', 'tab:goal008-' + str(i)) for i, family in enumerate(FAMILIES, 1))
    error_metrics = ('absolute_occupancy_iou', 'aligned_occupancy_iou', 'aligned_material_aware_iou', 'aligned_material_set_f1',
                     'aligned_total_normalized_repair_cost', 'aligned_add_count', 'aligned_delete_count', 'aligned_replace_count')
    errors = [[metric.replace('_', ' '), *[value for b in BUILDERS for value in (
        number(groups[b, 'O', metric]['equal_building_mean']),
        str(groups[b, 'O', metric]['valid_rows']) + '/' + str(groups[b, 'O', metric]['valid_buildings']))]] for metric in error_metrics]
    error_tex = table(['O metric', 'OpenAI mean', 'Slots / buildings', 'Anthropic mean', 'Slots / buildings'], errors,
        'O error structure from the frozen strict evaluator, valid-only equal-building means. Material-set F1 compares sets; material-aware voxel IoU also requires the correct material at matching coordinates. Add/delete/replace are operation counts, not time. Denominators are the O rows of the main table; alignment optimizes occupancy translation, with material identities unchanged.', 'tab:goal008-error-structure')
    exploratory = []
    for policy in POLICIES:
        for target in TARGETS:
            for association in ASSOCIATIONS:
                records = [r for r in associations['associations'] if (r['outcome_policy'], r['outcome_target'], r['association']) == (policy, target, association)]
                rows = [[NAMES.get(r['description_provider'], 'All'), COMPACT_FEATURES[r['feature']], number(r['coefficient']), interval(r['ci95']),
                    r['counts']['analysis_pairs'], r['counts']['analysis_buildings'], r['counts']['missing_outcome'],
                    f"{r['bootstrap_valid']}/{r['bootstrap_planned']}", STATUS_LABELS[r['status']]] for r in records]
                exploratory.append(table(['Desc.', 'Component', 'Coef.', '95% CI', 'Pairs', 'Bldgs', 'Miss. Y', 'Boots', 'Status'], rows,
                    PROXY_LABEL + '. ' + NAMES[target] + '; ' + policy.replace('_', ' ') + '; ' + ASSOCIATION_NAMES[association] + '. All planned strata retained. Strict/coarse material entries are F1; dimensions are relative accuracy; completeness measures typed fields. OK: estimate with cluster interval; CI?: estimate without eligible interval; C-X/Y/XY: constant proxy/outcome/both; 1/cluster: one description per building; N<2: insufficient pairs; Numeric?: undefined. Boots are finite/planned replicates. No p-values. Full statuses, tie/missing and interval diagnostics remain in the unchanged exploratory_associations_360.csv.',
                    'tab:goal008-proxy-' + str(len(exploratory) + 1)))
    return {'main_results.tex': main_tex, 'ablations.tex': ablation_tex, 'failures.tex': failure_tex,
            'stats_appendix.tex': appendix, 'error_structure.tex': error_tex, 'exploratory_appendix.tex': ('\n' + r'\clearpage' + '\n').join(exploratory)}, {
                'main_rows': main_exact, 'failure_rows': failure_exact, 'frozen_comparisons': results['comparisons'],
                'all_exploratory_associations': associations['associations'], 'error_metrics': error_metrics,
                'error_metric_rows': [groups[b, 'O', metric] for metric in error_metrics for b in BUILDERS]}


def scatter_rows(joined):
    """Display-only arithmetic, same complete-pair policy as the stored analysis."""
    complete = []
    for row in joined:
        values = [row['builders'][b]['valid_only'] for b in BUILDERS]
        x = row['scores']['composite']
        if x is None or any(v is None for v in values):
            continue
        complete.append({'scene_id': row['scene_id'], 'description_provider': row['description_provider'],
                         'input_id': row['input_id'], 'source_sha256': row['source_sha256'],
                         'composite': _finite(x), 'both_valid_mean_iou': sum(map(_finite, values)) / 2})
    by_scene = defaultdict(list)
    for row in complete:
        by_scene[row['scene_id']].append(row)
    for rows in by_scene.values():
        mean_x = sum(r['composite'] for r in rows) / len(rows)
        mean_y = sum(r['both_valid_mean_iou'] for r in rows) / len(rows)
        for row in rows:
            row['centered_composite'] = row['composite'] - mean_x if len(rows) >= 2 else None
            row['centered_mean_iou'] = row['both_valid_mean_iou'] - mean_y if len(rows) >= 2 else None
    return complete


def plot_figures(results, associations, output, *, fixture):
    import numpy as np
    os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': VERSION, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#0072B2', '#D55E00', '#009E73']
    def save(fig, name):
        for extension in ('png', 'svg'):
            fig.savefig(output / (name + '.' + extension), dpi=300, bbox_inches='tight',
                        metadata={'Software': VERSION} if extension == 'png' else {'Date': None, 'Creator': VERSION})
        plt.close(fig)
    groups = _groups(results)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.34, wspace=.46)
    labels = ['Geometry\n(absolute)', 'Geometry\n(aligned)', 'Exact mat.\n(aligned)']
    metrics = ('absolute_occupancy_iou', 'aligned_occupancy_iou', 'aligned_material_aware_iou')
    truncations = []
    for index, builder in enumerate(BUILDERS):
        axes[0].bar(np.arange(3) + (index - .5) * .32, [groups[builder, 'O', m]['equal_building_mean'] for m in metrics],
                    width=.32, color=colors[index], label=NAMES[builder], hatch='' if index == 0 else '//')
        e = groups[builder, 'E', 'aligned_occupancy_iou']
        axes[1].bar(np.arange(2) + (index - .5) * .32, [e['equal_building_mean'], e['failure0_equal_building_mean']],
                    width=.32, color=colors[index], hatch='' if index == 0 else '//')
        den = next(r for r in results['denominators']['groups'] if r['group_level'] == 'builder_condition' and r['provider'] == builder and r['condition'] == 'E')
        trunc = den['failure_kind_counts'].get('truncation', 0)
        truncations.append(trunc)
        axes[2].bar(index, trunc, color=colors[index], width=.55, hatch='' if index == 0 else '//')
        axes[2].annotate(f"{trunc}/{den['planned_slots']}", (index, trunc), xytext=(0, 4), textcoords='offset points', ha='center')
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel='Mean IoU', title='(a) O: geometry/material')
    axes[1].set(xticks=[0, 1], xticklabels=['Valid-only', 'F0'], ylim=(0, 1.06), ylabel='Mean aligned IoU', title='(b) E: valid-only / F0')
    e_ns = '; '.join(NAMES[b] + ' ' + str(groups[b, 'E', 'aligned_occupancy_iou']['valid_rows']) + '/' + str(groups[b, 'E', 'aligned_occupancy_iou']['planned_slots']) for b in BUILDERS)
    axes[1].text(.5, -.39, 'Valid slots\n' + e_ns, transform=axes[1].transAxes, ha='center', fontsize=11)
    axes[2].set(xticks=[0, 1], xticklabels=[NAMES[b] for b in BUILDERS], ylabel='Truncated outputs', title='(c) E: truncation')
    axes[2].set_ylim(0, max(1, max(truncations)) * 1.35)
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .98), ncol=2, frameon=False)
    fig.text(.5, .01, 'Equal-building means; E valid selection changes denominators.\nF0 covers known model failures; operational unknowns remain missing.', ha='center', fontsize=11)
    if fixture:
        fig.suptitle('SYNTHETIC FIXTURE — NOT RESEARCH RESULTS', y=1.14)
    save(fig, 'figure_main_errors')
    points = scatter_rows(associations['joined_descriptions'])
    expected = {(r['association']): r for r in associations['associations'] if r['description_provider'] is None and r['feature'] == 'composite' and r['outcome_policy'] == 'valid_only' and r['outcome_target'] == 'both_builders_mean'}
    centered = [r for r in points if r['centered_composite'] is not None]
    if len(points) != expected['pooled_spearman']['counts']['analysis_pairs'] or len(centered) != expected['within_building_centered_pearson']['counts']['analysis_pairs']:
        raise ValueError('display complete-case count differs from stored statistics')
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    fig.subplots_adjust(left=.085, right=.98, bottom=.20, top=.72, wspace=.30)
    for provider, color, marker in zip(PROVIDERS, colors, ('o', '^', 's')):
        rows = [r for r in points if r['description_provider'] == provider]
        axes[0].scatter([r['composite'] for r in rows], [r['both_valid_mean_iou'] for r in rows], label=NAMES[provider],
                        color=color, marker=marker, s=20, alpha=.75, linewidths=.3, edgecolors='white')
        rows = [r for r in centered if r['description_provider'] == provider]
        axes[1].scatter([r['centered_composite'] for r in rows], [r['centered_mean_iou'] for r in rows],
                        color=color, marker=marker, s=20, alpha=.75, linewidths=.3, edgecolors='white')
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Composite attribute proxy', ylabel='Both-valid Builder mean aligned IoU', title=f'(a) Complete descriptions: n={len(points)}')
    axes[1].axhline(0, color='#999999', linewidth=.6); axes[1].axvline(0, color='#999999', linewidth=.6)
    axes[1].set(xlabel='Within-building centered composite', ylabel='Within-building centered mean IoU', title=f'(b) Within-building pairs: n={len(centered)}')
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle(('SYNTHETIC FIXTURE\n' if fixture else '') + PROXY_LABEL, fontsize=11, y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .90), ncol=3, frameon=False, title='Description provider')
    fig.text(.5, .01, 'Two Builders repeat each description; buildings are independent units.\nNo fitted model or p-value is added.', ha='center', fontsize=10)
    save(fig, 'figure_proxy_scatter')
    return {'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'scatter_points': points,
            'figures': ['figure_main_errors', 'figure_proxy_scatter'], 'fitted_models': 0, 'p_values_computed': 0,
            'nominal_minimum_font_pt_at_7inch_width': {'figure_main_errors': 11*7/10.5, 'figure_proxy_scatter': 10*7/9.4}}


def english_sections(data):
    r, a, gate, sampling = data['results'], data['associations'], data['gate'], data['sampling']
    groups = _groups(r); contrasts = {row['id']: row for row in r['comparisons']}
    mean = lambda b, c, m: number(groups[b, c, m]['equal_building_mean'])
    def effect(id):
        row = contrasts[id]
        return f"{number(row['mean_delta'])} (95% CI {interval(row['ci95'])}; Holm p={p_value(row['p_holm'])}; {row['common_valid_pair_count']} pairs, {row['buildings']} buildings)"
    overall = r['denominators']['overall']
    intro = ('The frozen analysis contains ' + str(overall['planned_slots']) + ' selected slots on ' + str(overall['independent_buildings']) + ' buildings: '
        + str(overall['valid_only_aligned_iou_slots']) + ' valid reconstructions, ' + str(overall['model_output_failure_slots']) + ' identified model-output failures, and '
        + str(overall['operational_unmeasurable_slots']) + ' operationally missing observation. Selected completed outcomes number ' + str(overall['completed_request_outcome_slots'])
        + '. Buildings, rather than descriptions or Builder calls, are the independent units. All aggregate means first average within a building, then weight buildings equally; valid counts are shown with each condition. The strict translation evaluator is server_v2.2_strict_translation.')
    geometry = (f"For O, aligned occupancy IoU is {mean('openai','O','aligned_occupancy_iou')} for OpenAI Builder and {mean('anthropic','O','aligned_occupancy_iou')} for Anthropic Builder. "
        f"Absolute IoU is {mean('openai','O','absolute_occupancy_iou')}/{mean('anthropic','O','absolute_occupancy_iou')}, while aligned exact-material-aware voxel IoU is only {mean('openai','O','aligned_material_aware_iou')}/{mean('anthropic','O','aligned_material_aware_iou')}. "
        f"Aligned material-set F1 is {mean('openai','O','aligned_material_set_f1')}/{mean('anthropic','O','aligned_material_set_f1')}; set overlap does not require the material to occur at the correct voxel. "
        f"Normalized repair cost is {mean('openai','O','aligned_total_normalized_repair_cost')}/{mean('anthropic','O','aligned_total_normalized_repair_cost')}. The add/delete/replace decomposition is provided in error_structure.tex. These are counts of voxel operations, not human effort or elapsed repair time.")
    jo = ('O is the saved v4 JSON-form visual description record, including human-readable summary, materials, dimensions, elements and rebuild hints. J is a reserialization that preserves the same typed JSON tree; it is not the historical Structured IR. '
          + 'J-O aligned IoU contrasts are OpenAI ' + effect('J-minus-O:openai:aligned_occupancy_iou') + ' and Anthropic ' + effect('J-minus-O:anthropic:aligned_occupancy_iou')
          + '. Neither provides evidence of improved IoU after the original family correction; absence of evidence is not equivalence. The OpenAI J-O dimension MAE contrast is '
          + effect('J-minus-O:openai:aligned_dimension_mae') + ', where a positive value is worse. These are separate generation calls, so this contrast does not isolate a formatting-only causal effect.')
    oracle = (f"P supplies partial GT attributes, whereas E supplies the complete discrete GT specification. P dimension MAE is {mean('openai','P','aligned_dimension_mae')}/{mean('anthropic','P','aligned_dimension_mae')} and its IoU is {mean('openai','P','aligned_occupancy_iou')}/{mean('anthropic','P','aligned_occupancy_iou')}. "
        'The paired P-O MAE changes are OpenAI ' + effect('P-minus-O:openai:aligned_dimension_mae') + ' and Anthropic ' + effect('P-minus-O:anthropic:aligned_dimension_mae')
        + '. The P-O IoU tests do not establish improved geometry; all effects and intervals are retained in ablations.tex. '
        + f"E valid-only IoU is {mean('openai','E','aligned_occupancy_iou')}/{mean('anthropic','E','aligned_occupancy_iou')}, but known-failure-zero IoU is "
        + '/'.join(number(groups[b,'E','aligned_occupancy_iou']['failure0_equal_building_mean']) for b in BUILDERS)
        + '. E truncation is ' + '/'.join(str(next(x for x in r['denominators']['groups'] if x['group_level']=='builder_condition' and x['provider']==b and x['condition']=='E')['failure_kind_counts'].get('truncation',0)) + ' of 60' for b in BUILDERS)
        + ' for OpenAI/Anthropic, respectively. On common-valid E buildings, the Anthropic-minus-OpenAI contrast is ' + effect('anthropic-minus-openai:E:aligned_occupancy_iou')
        + '. Unequal valid-only marginal means do not establish a Builder ranking. E is a conditional diagnostic, not a whole-population upper bound or a pure language-information intervention.')
    history = (f"The frozen provenance retains {gate['fixed_scope_attempts']} fixed-scope attempts and {gate['historical_calls']} historical calls. Slot 940 selects the explicitly approved successful same-payload authentication retry while retaining the original HTTP 401 attempt. Slot 942 remains HTTP 520 with null response, usage and reconstruction metrics; no IoU=0 is imputed. Both unknown-usage reservations remain intact. Budget-accounted USD {gate['budget_accounted_usd']} is not an invoice amount.")
    oldrq2 = ('The original conventional-description-quality question remains ' + RQ2_STATUS + '. Historical per-sample scores do not establish an exact match to the current 180 records: 40 of 60 Claude dimension records differ, and matched Gemini scores are absent. Historical means and v5 data are not substituted. This does not show that conventional description quality fails to predict reconstruction.')
    proxy = ('A separate, new CPU-derived attribute fidelity proxy was specified before these scores and associations were calculated. It uses strict material F1, coarse material F1, missing-safe dimension accuracy and typed field completeness, with fixed weights 0.30/0.25/0.25/0.20. All 180 descriptions have exact original-source, input and GT hash binding. Missing or invalid dimensions are penalized rather than filled from GT. This proxy is not conventional description quality or a human judgment. '
        'All 360 specified associations are retained, including nonestimable strata and unstable intervals; no p-value or Holm family was added. Two Builder outcomes repeat each description. Intervals resample 60 building clusters 2,000 times with PCG64 seed 20261016; this seed is not a decoding seed.')
    proxy_lines = []
    for policy in POLICIES:
        for target in TARGETS:
            for association in ASSOCIATIONS:
                row = next(x for x in a['associations'] if x['feature']=='composite' and x['description_provider'] is None and (x['outcome_policy'],x['outcome_target'],x['association'])==(policy,target,association))
                proxy_lines.append(f"{NAMES[target]}, {policy}, {ASSOCIATION_NAMES[association]}: coefficient {number(row['coefficient'])}, 95% CI {interval(row['ci95'])}, {row['counts']['analysis_pairs']} pairs/{row['counts']['analysis_buildings']} buildings; {row['status']} ({row['bootstrap_valid']}/{row['bootstrap_planned']} bootstrap estimates finite).")
    def composite_record(target, association):
        return next(x for x in a['associations'] if x['feature']=='composite' and x['description_provider'] is None
                    and (x['outcome_policy'],x['outcome_target'],x['association'])==('valid_only',target,association))
    def association_text(row):
        return f"{number(row['coefficient'])} (95% CI {interval(row['ci95'])}; {row['counts']['analysis_pairs']} descriptions, {row['counts']['analysis_buildings']} buildings)"
    proxy_results = ('For valid O reconstructions, the composite proxy has pooled Spearman associations of '
        + association_text(composite_record('openai','pooled_spearman')) + ' with OpenAI Builder IoU and '
        + association_text(composite_record('anthropic','pooled_spearman')) + ' with Anthropic Builder IoU. '
        + 'Using the mean of both Builders only when both are valid gives ' + association_text(composite_record('both_builders_mean','pooled_spearman')) + '. '
        + 'Within-building centered associations are OpenAI ' + association_text(composite_record('openai','within_building_centered_pearson'))
        + ', Anthropic ' + association_text(composite_record('anthropic','within_building_centered_pearson'))
        + ', and both-Builder mean ' + association_text(composite_record('both_builders_mean','within_building_centered_pearson')) + '. '
        + 'These exploratory associations do not validate conventional description quality or establish causality. All five components, rank-centered results, provider strata and failure0 sensitivity records remain in the separate exploratory appendix, including null and unstable intervals. '
        + 'Across the full fixed analysis grid, ' + str(a['status_counts'].get('ESTIMABLE', 0)) + ' records have point estimates and cluster intervals, '
        + str(a['status_counts'].get('ESTIMATE_WITHOUT_CLUSTER_CI', 0)) + ' retain estimates without an eligible cluster interval, '
        + str(sum(n for status, n in a['status_counts'].items() if status.startswith('CONSTANT_'))) + ' are undefined because of constant variables, and '
        + str(a['status_counts'].get('NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING', 0)) + ' provider-stratified within-building records are not estimable with one description per building.')
    sampling_text = (f"O/J comprise {sampling['O_attempts']}/{sampling['J_attempts']} attempts over 360 slots each; the additional O attempt is the approved slot 940 retry. These are single-output calls without controlled repeated sampling. No request explicitly fixes temperature, top-p, top-k, penalties or decoding seed. Available OpenAI response metadata report temperature 1 and top-p 1; Anthropic reports no such sampling fields, so its defaults remain unknown. Response metadata are not a guarantee of deterministic decoding. Fixed model identities and the full safe metadata distribution are retained in sampling_audit.json.")
    discussion = ('The low O geometry overlap and still lower exact-material-aware overlap indicate distinct geometric and material-placement errors in this measured reconstruction setting. A material appearing in the description is insufficient to place it correctly. P illustrates that improved dimensions need not recover voxel layout; E illustrates that supplying the full discrete target can make valid outputs much closer while leaving output-capacity failures. Neither intervention identifies how much information natural language causally loses.\n\n'
        'The exploratory proxy associations concern explicitly scored attributes from the same frozen descriptions used by O. They provide a reproducible association analysis, not independent validation of the proxy. Material and dimension proxies share GT with reconstruction metrics, so construction effects can contribute to associations. Within-building centering removes additive building offsets for the complete pairs used; it does not remove all confounding or establish a causal quality effect. All fixed components, Builder targets, provider strata, failure0 sensitivities and nonestimable results remain visible to avoid choosing only favorable associations.\n\n'
        'The revised question structure below is post-hoc evidence-based reframing for the paper. It must not be described as preregistered. The original 20 comparisons retain their original families and status; the new 360-record exploratory grid was fixed before its computation and is separate from those comparisons.')
    limitations = ('The independent sample is at most 60 buildings, from a fixed v4 cohort, with fixed descriptions, prompts, Builder versions, schema and evaluator. The three image-to-description providers (OpenAI, Anthropic and Gemini) have a different role from the two text-to-reconstruction Builders (OpenAI and Anthropic). Repeated Builder outcomes and provider descriptions do not create 360 independent buildings. Valid-only selection, especially under E, changes the estimand; F0 is a bounded-IoU sensitivity for known completed failures only and cannot recover unknown operational outcomes. Failure MAE and repair remain unmeasured.\n\n'
        + sampling_text + '\n\n' + oldrq2 + '\n\n'
        + 'The new proxy scores top-level structured attributes rather than every claim in summary or hints. Dominant-material selection, lexical strictness, coarse categories, storage-bbox dimensions, completeness ties, chosen weights and missing-value penalties are measurement choices rather than validated human-quality judgments. Shared GT, exploratory multiplicity, missingness and selected valid pairs constrain interpretation. Bootstrap dependence correction does not address these validity limits. One-description-per-building provider strata cannot support within-building correlations; constants and unstable bootstrap intervals are reported explicitly. No human experiment validates utility or repair time.\n\n'
        + 'The current fixed 960-slot implementation and evaluator are not identical to historical ARR 200-building, earlier 1,200-run or v5 studies; those observations and scores are not pooled. Example scenes follow a fixed rule but remain illustrations, not population estimates. Additional CPU work can review claim-to-table provenance and alternative transparent proxy assumptions as separately labeled sensitivity analyses; it cannot resolve conventional RQ2 without properly matched, validated quality evidence.')
    return {'Results.md': '# Results\n\n' + '\n\n'.join((intro, geometry, jo, oracle, history, '## Historical conventional-quality RQ2\n\n' + oldrq2, '## Separate exploratory CPU attribute proxy\n\n' + proxy, proxy_results)) + '\n',
            'Discussion.md': '# Discussion\n\n' + discussion + '\n', 'Limitations.md': '# Limitations\n\n' + limitations + '\n',
            'exploratory_proxy_summary.md': '# ' + PROXY_LABEL + '\n\n' + proxy + '\n\n' + '\n\n'.join(proxy_lines) + '\n\nAll 5 features, all 3 description-provider strata, all targets and both outcome policies appear in exploratory_appendix.tex and the unchanged 360-row CSV; their tie/missing/constant diagnostics remain in the source JSON. Status counts: ' + json.dumps(a['status_counts'], sort_keys=True) + '.\n',
            'RQ_REFRAMING.md': '# Proposed post-hoc question reframing\n\nThis organization is a new post-hoc interpretation of the available evidence, not a change to the predeclared 20 comparisons.\n\n1. How accurately can the fixed visual-description records support voxel reconstruction?\n2. How do information-preserving JSON reserialization and separate GT diagnostic inputs behave? J tests reserialization through separate calls; P/E add GT information.\n3. What exploratory associations connect source-bound attribute fidelity proxies with reconstruction? This does not replace the untestable conventional-quality RQ2.\n'}


def _captions(example_info):
    first = r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure_main_errors.png}
\caption{Frozen Goal007 errors: (a) O absolute/aligned geometry and aligned exact-material-aware voxel IoU; (b) E valid-only and known-failure-zero aligned IoU; (c) E truncation. Means weight buildings equally, with valid selection changing denominators. F0 includes only identified completed model failures, never slot 942 HTTP 520. E valid-only values are conditional, not population upper bounds. No new inference is performed.}
\label{fig:goal008-main-errors}
\end{figure*}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure_proxy_scatter.png}
\caption{Exploratory CPU attribute proxy---not conventional description quality. Composite versus mean aligned IoU for descriptions with both Builders valid, and the same complete pairs centered within each building. Color and marker denote description provider. The two Builder outcomes are repeated observations, not independent descriptions; uncertainty in the separate tables resamples buildings. No fitted line or p-value is added. Shared GT and complete-case selection limit interpretation.}
\label{fig:goal008-proxy-scatter}
\end{figure*}
'''
    if example_info:
        first += '\n% Third body figure: three separately verified reconstruction-example rows.\n' + r'\input{examples_figure.tex}' + '\n'
    else:
        first += '\n% Third body figure is supplied separately as examples_figure.tex after verified example rendering.\n'
    return first


def check_latex(output, *, examples_ready):
    """Use an existing compiler only; disabled shell escape and no installation."""
    sections = ['main_results.tex', 'ablations.tex', 'failures.tex', 'stats_appendix.tex', 'error_structure.tex', 'exploratory_appendix.tex', 'manuscript_sections.tex']
    content = '\n'.join((output / name).read_text() for name in sections)
    if content.count(r'\begin{table*}') != content.count(r'\end{table*}'):
        raise ValueError('unbalanced table environments')
    columns = None
    for line in content.splitlines():
        if line.startswith(r'\begin{tabular}{'):
            columns = len(line.removeprefix(r'\begin{tabular}{').removesuffix('}'))
        elif columns is not None and line.endswith(r' \\'):
            # Escaped literal ampersands are content, not column separators.
            if len(line.replace(r'\&', '').split(' & ')) != columns:
                raise ValueError('TeX table column count mismatch')
        elif line.startswith(r'\end{tabular}'):
            columns = None
    wrapper = '\n'.join([r'\documentclass[twocolumn]{article}', r'\usepackage[T1]{fontenc}', r'\usepackage[utf8]{inputenc}',
                         r'\usepackage{graphicx,booktabs}', r'\begin{document}',
                         *[r'\input{' + name + '}' for name in sections],
                         r'\input{figure_captions.tex}' if examples_ready else r'\includegraphics[width=\textwidth]{figure_proxy_scatter.png}',
                         r'\end{document}', ''])
    _new(output / 'compile_check.tex', wrapper)
    executable = shutil.which('pdflatex')
    if executable is None:
        return {'status': 'NOT_TESTED_COMPILER_UNAVAILABLE', 'compiler': None,
                'static_checks': 'escaped table cells; matched table environments; compact rows and three separate confirmatory families', 'packages_required': ['graphicx', 'booktabs']}
    try:
        result = subprocess.run([executable, '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', 'compile_check.tex'],
                                cwd=output, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {'status': 'FAILED_TIMEOUT', 'compiler': executable}
    _new(output / 'latex_compilation.log', result.stdout.decode('utf-8', errors='replace') + result.stderr.decode('utf-8', errors='replace'))
    return {'status': 'PASS' if result.returncode == 0 else 'FAILED', 'compiler': executable, 'returncode': result.returncode}


def render_publication(goal007_directory, phase1_directory, audit_directory, output_directory, *, examples_manifest=None):
    started = time.monotonic()
    output = Path(output_directory).absolute()
    if output.exists():
        raise ValueError('new publication directory required; preserve previous artifacts')
    data = load_inputs(goal007_directory, phase1_directory, audit_directory)
    example_info = None
    if examples_manifest:
        path = _regular(examples_manifest); example_info = read_json(path)
        if example_info.get('status') != 'RENDERED' or example_info.get('input_kind') != data['results']['input_kind'] or len(example_info['examples']) != 3:
            raise ValueError('three verified examples required')
        data['input_files'][str(path)] = sha256(path)
        for record in example_info['files']:
            file = _product(path.parent, record['path'], record)
            data['input_files'][str(file)] = sha256(file)
    docs, numeric = build_tables(data['results'], data['associations'])
    docs.update(english_sections(data))
    if data['results']['input_kind'] == 'fixture':
        docs = {k: ('% SYNTHETIC FIXTURE — NOT RESEARCH RESULTS\n' if k.endswith('.tex') else '> SYNTHETIC FIXTURE — NOT RESEARCH RESULTS\n\n') + v for k, v in docs.items()}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in docs.items():
        _new(output / name, content)
    sections = []
    for name in ('Results.md', 'Discussion.md', 'Limitations.md'):
        for paragraph in docs[name].split('\n\n'):
            if paragraph.startswith('# '):
                sections.append(r'\section{' + tex(paragraph[2:].strip()) + '}')
            elif paragraph.startswith('## '):
                sections.append(r'\subsection{' + tex(paragraph[3:].strip()) + '}')
            else:
                sections.append(tex(paragraph.strip()))
    _new(output / 'manuscript_sections.tex', '\n\n'.join(sections) + '\n')
    _new(output / 'publication_numbers.json', _json(numeric))
    _new(output / 'frozen_comparisons_20.json', _json(data['results']['comparisons']))
    # Exact copy, preserving every coefficient, missing value and stored diagnostic.
    with (output / 'exploratory_associations_360.csv').open('xb') as stream:
        stream.write(Path(data['exploratory_csv_path']).read_bytes())
    _new(output / 'figure_captions.tex', _captions(example_info))
    figures = plot_figures(data['results'], data['associations'], output, fixture=data['results']['input_kind'] == 'fixture')
    _new(output / 'proxy_scatter_points.json', _json(figures.pop('scatter_points')))
    if example_info:
        for record in example_info['files']:
            if record['path'] in {'example_01.png', 'example_02.png', 'example_03.png', 'examples_figure.tex'}:
                with (output / record['path']).open('xb') as stream:
                    stream.write((Path(examples_manifest).parent / record['path']).read_bytes())
    latex = check_latex(output, examples_ready=bool(example_info))
    _new(output / 'latex_validation.json', _json(latex))
    after = {path: sha256(path) for path in data['input_files']}
    if after != data['input_files']:
        raise ValueError('publication input changed while rendering; artifacts are incomplete')
    files = {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(output.iterdir()) if p.is_file()}
    report = {'status': 'PUBLICATION_RENDERED', 'version': VERSION, 'input_kind': data['results']['input_kind'],
              'phase1_code_commit': data['phase1']['created_code_commit'], 'publication_code_sha256': sha256(__file__),
              'input_files': after, 'input_files_before_after_unchanged': True, 'products': files,
              'original_comparisons': 20, 'original_families': FAMILIES, 'exploratory_records': 360,
              'historical_RQ2_status': RQ2_STATUS, 'new_proxy_label': PROXY_LABEL, 'figures': figures,
              'examples_included': example_info['examples'] if example_info else [], 'latex_validation': latex,
              'research_api_requests_sent': 0, 'api_key_queries': 0, 'human_experiments': 0,
              'new_inference_runs': 0, 'source_mutations': 0, 'checkpoint_mutations': 0,
              'performance': {'seconds': time.monotonic() - started, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    _new(output / 'RUN.json', _json(report))
    digest = sha256(output / 'RUN.json')
    _new(output / 'RUN.sha256', digest + '  RUN.json\n')
    return {'status': report['status'], 'run_path': str(output / 'RUN.json'), 'run_sha256': digest,
            'latex_status': latex['status'], 'products': len(files), 'research_api_requests_sent': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('goal007-directory', 'phase1-directory', 'audit-directory', 'output-directory'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--examples-manifest')
    args = parser.parse_args(argv)
    result = render_publication(args.goal007_directory, args.phase1_directory, args.audit_directory,
                                args.output_directory, examples_manifest=args.examples_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
