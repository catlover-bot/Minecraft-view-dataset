"""Publication artifacts from audited Goal007 statistics; no inference or API I/O.

The supplied statistics are the sole numerical source. Scientific figures use
matplotlib's noninteractive Agg backend. Outputs are created once, never replaced.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

ANALYSIS_LABEL = 'fixed60 analysis with one operational missing observation'
RQ2_UNAVAILABLE = 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES'
BUILDERS = ('openai', 'anthropic')
CONDITIONS = ('O', 'J', 'P', 'E')
METRICS = ('aligned_occupancy_iou', 'aligned_dimension_mae')
FAMILIES = {'primary_builder_valid_pair': 8, 'secondary_lossless_J_minus_O': 4,
            'diagnostic_oracle_minus_original': 8}
NAMES = {'openai': 'OpenAI', 'anthropic': 'Anthropic', 'gemini': 'Gemini'}
METRIC_LABELS = {'aligned_occupancy_iou': 'Aligned occupancy IoU',
                 'aligned_dimension_mae': 'Aligned dimension MAE (blocks)'}
FIGURES = ('figure_main', 'figure_description_builders', 'figure_comparisons')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'


def _number(value, precision=4):
    if value is None:
        return 'NA'
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('displayed statistics must be finite numbers or null')
    return f'{value:.{precision}f}'


def _p(value):
    return 'NA' if value is None else f'{value:.6g}'


def _ci(value):
    return 'NA' if value is None else '[' + ', '.join(_number(v) for v in value) + ']'


def _tex(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in str(value))


def _table(headers, rows, caption, label):
    lines = [r'\begin{table*}[t]', r'\centering', r'\scriptsize',
             r'\caption{' + _tex(caption) + '}', r'\label{' + label + '}',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
             ' & '.join(_tex(v) for v in headers) + r' \\', r'\midrule']
    lines += [' & '.join(_tex(v) for v in row) + r' \\' for row in rows]
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}}', r'\end{table*}', ''])


def _csv(rows):
    # Preserve scalar precision; JSON cells retain nested provenance/counts.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False)
                         if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return handle.getvalue()


def _lookup(results):
    return {(r['group_level'], r['provider'], r['condition'], r.get('description_model'), r['metric']): r
            for r in results['per_group']}


def _contrast(row):
    if row['a_provider'] != row['b_provider']:
        return 'Anthropic - OpenAI / ' + row['a_condition']
    return row['b_condition'] + ' - ' + row['a_condition'] + ' / ' + NAMES[row['a_provider']]


def _validate(results, context, rq2):
    if results.get('analysis_label') != ANALYSIS_LABEL:
        raise ValueError('Goal007 missing-observation analysis label required')
    kind = results.get('input_kind')
    if kind not in {'fixture', 'verified_sources'} or context.get('status') != 'PASS' or context.get('input_kind') != kind:
        raise ValueError('matching passing audit context required')
    if rq2.get('status') != RQ2_UNAVAILABLE:
        raise ValueError('renderer currently requires the audited unavailable matched-score RQ2 status')
    if results.get('research_api_requests_sent') != 0:
        raise ValueError('Goal007 analysis must send zero research requests')
    comparisons = results['comparisons']
    if len(comparisons) != 20 or len({r['id'] for r in comparisons}) != 20:
        raise ValueError('exactly twenty distinct predeclared contrasts required')
    actual = {family: sum(r['family_id'] == family for r in comparisons) for family in FAMILIES}
    if actual != FAMILIES or any(r['family_id'] not in FAMILIES for r in comparisons):
        raise ValueError('predeclared Holm families must remain 8/4/8')
    for row in comparisons:
        if row['buildings'] > 60 or row['metric'] not in METRICS:
            raise ValueError('invalid independent building count or primary metric')
        for name in ('mean_delta', 'p_two_sided', 'p_holm'):
            _number(row[name])
        if row['metric'] == 'aligned_dimension_mae' and row.get('common_failure0_pair_count') is not None:
            raise ValueError('MAE cannot have failure0 observations')
    overall = results['denominators']['overall']
    if kind == 'verified_sources':
        if (overall['planned_slots'] != 960 or overall['independent_buildings'] != 60
                or overall['operational_unmeasurable_slots'] != 1
                or context.get('fixed_scope_attempts') != 961 or context.get('historical_calls') != 991
                or context.get('operational_error_attempts') != 2):
            raise ValueError('real fixed60 slot/attempt denominators must be verified')
        holds = context.get('unknown_usage_holds', [])
        if len(holds) != 2:
            raise ValueError('both unknown-usage holds must be reported')
        for name in ('cost_rows', 'failure_taxonomy', 'error_rows', 'source_proofs'):
            if not isinstance(context.get(name), list) or not context[name]:
                raise ValueError('real rendering requires nonempty ' + name)
        if not isinstance(context.get('reproduction_command'), str) or not context['reproduction_command'].strip():
            raise ValueError('exact CPU reproduction command required')
    _json(results)
    _json(context)


def _table_outputs(results):
    lookup = _lookup(results)
    main = []
    for provider in BUILDERS:
        for condition in CONDITIONS:
            for metric in METRICS:
                r = lookup[('builder_condition', provider, condition, None, metric)]
                main.append([NAMES[provider], condition, 'IoU' if metric == METRICS[0] else 'MAE',
                             f"{r['valid_rows']}/{r['planned_slots']}", r['valid_buildings'], _number(r['equal_building_mean']),
                             _number(r.get('failure0_equal_building_mean')), r.get('failure0_eligible_rows') if r.get('failure0_eligible_rows') is not None else 'NA'])
    contrasts = []
    ablations = []
    for r in results['comparisons']:
        comp = r.get('all_attempted_failure0_companion') or {}
        row = [r['family_id'].split('_')[0], _contrast(r), 'IoU' if r['metric'] == METRICS[0] else 'MAE',
               f"{r['common_valid_pair_count']}/{r['buildings']}", _number(r['mean_delta']), _ci(r['ci95']),
               _p(r['p_two_sided']), _p(r['p_holm']),
               f"{r['common_failure0_pair_count']}/{r['common_failure0_buildings']}" if r.get('common_failure0_pair_count') is not None else 'NA',
               _number(comp.get('mean_delta'))]
        contrasts.append(row)
        if r['family_id'] != 'primary_builder_valid_pair':
            ablations.append(row)
    failures = []
    for r in results['denominators']['groups']:
        if r['group_level'] == 'builder_condition':
            failures.append([NAMES[r['provider']], r['condition'], r['planned_slots'], r['completed_request_outcome_slots'],
                             r['valid_only_aligned_iou_slots'], r['model_output_failure_slots'],
                             r['operational_unmeasurable_slots'], r['unknown_usage_slots'], r['bounded_iou_eligible_slots']])
    headers = ['Family', 'B - A contrast', 'Metric', 'Pairs/buildings', 'Delta', '95% CI', 'Raw p', 'Holm p', 'F0 pairs/buildings', 'F0 delta']
    return {
      'table_main.tex': _table(['Builder', 'Condition', 'Metric', 'Valid/planned slots', 'Buildings', 'Mean', 'F0 IoU mean', 'F0 eligible slots'], main,
        'Equal-building means. IoU higher is better; MAE lower is better. Valid-only denominators are metric-specific. F0 is an auxiliary bounded-IoU score with known completed model-output failures set to zero; operational missingness remains excluded. NA is not zero.', 'tab:goal007-main'),
      'table_ablation.tex': _table(headers, ablations,
        'Predeclared J-O and oracle-O paired contrasts. J is lossless JSON reserialization of O. P and E add GT information and are diagnostic, not causal estimates of pure language-information loss. F0 is descriptive only.', 'tab:goal007-ablation'),
      'table_failures.tex': _table(['Builder', 'Condition', 'Slots', 'Completed outcomes', 'Valid IoU', 'Model failures', 'Operational missing', 'Unknown usage slots', 'F0 eligible'], failures,
        'Selected-slot outcomes. Completed outcomes include known model-output failures. The HTTP401 at global940 remains in attempt history; its authorized successful retry supplies the selected observation. HTTP520 at global942 remains unmeasured. Attempt history and unknown-usage holds are reported separately.', 'tab:goal007-failures'),
      'table_statistics.tex': _table(headers, contrasts,
        'All twenty predeclared comparisons; effects are B-A. Positive IoU is better for B; positive MAE is worse for B. Bootstrap 95% intervals and two-sided sign-flip p-values use buildings as independent units; Holm corrections are within the frozen families of 8, 4 and 8. F0 is auxiliary bounded IoU and supplies no additional significance claim.', 'tab:goal007-statistics'),
    }


def _main_markdown(results):
    lookup = _lookup(results)
    lines = ['| Builder | Condition | IoU (valid slots/buildings) | MAE (valid slots/buildings) | F0 IoU (eligible slots) |',
             '|---|---|---:|---:|---:|']
    for provider in BUILDERS:
        for condition in CONDITIONS:
            i = lookup[('builder_condition', provider, condition, None, METRICS[0])]
            m = lookup[('builder_condition', provider, condition, None, METRICS[1])]
            lines.append(f"| {NAMES[provider]} | {condition} | {_number(i['equal_building_mean'])} ({i['valid_rows']}/{i['valid_buildings']}) | {_number(m['equal_building_mean'])} ({m['valid_rows']}/{m['valid_buildings']}) | {_number(i['failure0_equal_building_mean'])} ({i['failure0_eligible_rows']}) |")
    return '\n'.join(lines)


def _contrast_markdown(results):
    lines = ['| Family | Contrast | Metric | Pairs/buildings | Delta [95% CI] | Raw p | Holm p |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in results['comparisons']:
        lines.append(f"| {r['family_id']} | {_contrast(r)} | {'IoU' if r['metric'] == METRICS[0] else 'MAE'} | {r['common_valid_pair_count']}/{r['buildings']} | {_number(r['mean_delta'])} {_ci(r['ci95'])} | {_p(r['p_two_sided'])} | {_p(r['p_holm'])} |")
    return '\n'.join(lines)


def _documents(results, context, rq2):
    d = results['denominators']['overall']
    n = d['planned_slots']
    valid = d['valid_only_aligned_iou_slots']
    failures = d['model_output_failure_slots']
    completed = d['completed_request_outcome_slots']
    eligible = d['bounded_iou_eligible_slots']
    kind = 'SYNTHETIC FIXTURE — NOT RESEARCH RESULTS\n\n' if results['input_kind'] == 'fixture' else ''
    history = f"{context.get('fixed_scope_attempts', 'UNSPECIFIED')} fixed-scope attempts and {context.get('historical_calls', 'UNSPECIFIED')} historical calls"
    sig = {f: sum(r['p_holm'] is not None and r['p_holm'] < .05 for r in results['comparisons'] if r['family_id'] == f) for f in FAMILIES}
    main, contrasts = _main_markdown(results), _contrast_markdown(results)
    inference = results['inference']
    en = kind + f"""# Goal007 ARR Results / Discussion / Limitations draft

Analysis label: **{ANALYSIS_LABEL}**.

## Results and RQ1: reconstruction from saved image descriptions

The frozen design contains {n} slots across {d['independent_buildings']} buildings, two Builders, and three saved description providers. O and J contain 360 slots each; P and E contain 120 each. Oracle conditions are not replicated across description providers. The selected observations comprise {completed} identified completed outcomes, including {failures} model-output failures, and {d['operational_unmeasurable_slots']} operationally unmeasurable slot. There are {valid} valid aligned-IoU observations and {eligible} observations eligible for the auxiliary bounded-IoU analysis. These counts are not interchangeable with {history}. This analysis sent zero research requests.

{main}

All means above weight buildings equally after averaging eligible observations within each building. IoU is on [0,1], with larger values better; dimension MAE is in blocks, with smaller values better. Secondary geometry, material-aware IoU, material-set F1, repair cost, and addition/deletion/replacement counts are available with metric-specific denominators in `per_group.csv`. The saved description provider and the Builder are different experimental roles; all twelve description-provider × Builder × O/J combinations are included in `per_group.csv` and `figure_description_builders`.

Global940 uses the explicitly approved manual authentication retry after the original HTTP401. The original attempt and its unknown-usage reservation remain in history. Global942 (building_085, J, OpenAI Builder) had HTTP520 with unknown usage and remains `operational_unmeasurable`: its response, usage, and metrics are null. It is not a completed quality observation or a model-output failure and contributes no zero to any primary or failure0 analysis. Both unknown-usage holds remain in the separate accounting audit; calculated usage cost is not an invoice.

## RQ3 and the predeclared paired comparisons

J is a lossless JSON reserialization of the original O description, not the historical Structured IR intervention. P provides partial GT attributes and E provides exact discrete GT. Oracle contrasts include Builder capability and representation/execution effects; they do not identify a pure causal effect of language-information loss.

{contrasts}

The three families contain 8, 4, and 8 comparisons. At Holm-adjusted p < 0.05, respectively {sig['primary_builder_valid_pair']}, {sig['secondary_lossless_J_minus_O']}, and {sig['diagnostic_oracle_minus_original']} comparisons pass that threshold; the table retains every contrast regardless of significance. Effects are B−A on common-valid pairs, averaged within buildings and then equally across buildings. Positive IoU differences favor B; positive MAE differences indicate greater error for B. Statistical non-significance alone does not establish equivalence. Auxiliary failure0 effects and eligible pair/building counts are in `comparisons_20.csv` and the LaTeX tables; they are bounded-IoU descriptive companions, not additional significance tests.

## RQ2: description-quality scores versus reconstruction

**{rq2['status']}**. Existing historical per-sample scores were identified, but their source provenance does not establish strict correspondence to all current frozen v4 descriptions. Some historical description fields differ and Gemini matched scores are absent. Historical model means, old v5 records, and three-model-average correlations are not substituted. No new RQ2 correlation or bootstrap is reported. See `rq2_matched_scores_audit.json` for searched paths, content hashes, and exclusion evidence.

## Missing-observation sensitivity and limitations

The primary analysis excludes operationally missing measurements from common-valid pairs while retaining the slot in the design. `sensitivity_missing942.json` separately records affected contrasts and exploratory aligned-IoU endpoint bounds using hypothetical values 0 and 1. These endpoints are possible-value assumptions, not imputations, observed failures, confidence intervals, or new p-values; no post-hoc significance conclusion is drawn. Unbounded MAE has no analogous finite [0,1] bound.

The independent sample is at most 60 buildings, not 960 independent trials. Selection on valid output can alter the main estimand; bounded-IoU companions show the effect of known output failures but cannot resolve unknown operational outcomes. The experiment fixes models, prompts, saved descriptions, GT, and evaluator implementation. It supports claims about this measured voxel reconstruction setting; broader scenes, human utility, language understanding, and model populations require separate evidence. Equal-building weights prevent description-provider repetitions from inflating independent N. Operational missingness, selected approved retry provenance, model-output failures, unavailable strictly matched quality scores, and oracle interpretation limit any claim that language descriptions generally preserve sufficient information.

CPU-only priorities are to inspect the existing metric/failure breakdowns and paired interval widths, review existing saved descriptions and reconstruction examples without altering observations, and, if a new scoring protocol is later specified, construct provenance-bound description-quality evidence as a separately labeled exploratory analysis. No additional generation is needed for those audits.
"""
    ja = kind + f"""# Goal007：最終統合解析の要点

分析ラベルは **{ANALYSIS_LABEL}**。固定60建築・{n} slots（O/J各360、P/E各120）を統合した。selected完了応答は{completed}、そのうちモデル出力失敗{failures}。aligned IoUのvalid観測は{valid}、bounded IoUのfailure0適格は{eligible}、運用上の未測定slotは{d['operational_unmeasurable_slots']}。固定対象の試行履歴{context.get('fixed_scope_attempts', '未指定')}件と歴史的送信{context.get('historical_calls', '未指定')}件を区別し、本解析の研究API送信は0回。

{main}

平均は建築内で集約してから建築を等重みで平均した。表のvalid slots/buildingsは指標固有の分母であり、独立Nは最大60。IoUは高いほど良く、MAEは低いほど良い。材質・repair・追加/削除/置換の詳細と説明文provider×Builderの全組合せは `per_group.csv` に記録した。

global940は元HTTP401と予約を消さず、明示承認された認証再試行の成功を最終観測に選択した。global942のHTTP520は応答・usage・metrics=nullの運用欠測であり、モデル失敗やIoU=0へ変換していない。元401と補完520のunknown usage holdは2件とも保持する。会計の計算額を実請求額とは呼ばない。

事前指定20比較を3family（8/4/8）内でHolm補正した。p_holm<0.05はBuilder差{sig['primary_builder_valid_pair']}件、J−O差{sig['secondary_lossless_J_minus_O']}件、oracle−O差{sig['diagnostic_oracle_minus_original']}件。効果・方向・95%区間・p_raw/p_holm・共通valid pair/building数を全件掲載し、非有意差も省略しない。非有意は同等性の証明ではない。

JはOの情報保存型JSON再シリアライズであり、旧Structured IRではない。P/EはGT情報を追加する診断条件で、純粋な言語情報損失の因果効果とは解釈できない。RQ2は **{rq2['status']}**。旧per-sample品質値は存在するが、現descriptionとの厳密な同一性を証明できず、一部は実際に内容が異なる。旧モデル平均・v5の流用や新規相関の計算はしていない。

HTTP520欠測のIoU [0,1]仮定は別のexploratory boundに限定し、p値を追加しない。限界は固定60建築、valid選択、モデル出力失敗、運用欠測1、品質スコア対応不足、oracleの解釈範囲である。追加APIなしの優先作業は、既存failure/metricの内訳確認、paired CIの精度確認、既存例の出典付き点検、新たに定義されたCPU品質評価を別の探索的証拠として整備することである。
"""
    audit = kind + f"""# Goal007 audit and reproduction

Status: source audit PASS as supplied in `render_context.json`; analysis label **{ANALYSIS_LABEL}**. Rendering consumes the frozen `results.json`; it does not rerun inference, the evaluator, API clients, key lookup, or GitHub writes.

## Sources and classification

`render_context.json` records source proofs, costs, unknown-usage holds, and the reproduction command supplied by integration. `rq2_matched_scores_audit.json` records the separate matched-score investigation. The source plans, immutable hashes, restored releases/assets, ledger attempts, raw/parsed/voxel references, evaluator caches, and final source inventories are documented by the integration MANIFEST and audit receipts, not inferred from successful figure rendering.

Final slots are separate from attempts: {n} selected slots; {history}; zero new research requests. Global940 selects the explicitly approved manual retry while preserving original401 attempt provenance and hold. Global942 is operationally unmeasurable HTTP520, with null response, usage, and metrics. Completed-request status requires identified completion and known usage; a received HTTP response alone is insufficient. Model-output parse/validation/execution/truncation failures are retained. Only these known completed failures may receive zero in auxiliary bounded IoU. Failed MAE/repair stay null. Operational unknowns never enter failure0. Actual invoices remain unknown.

## Exact metrics and all denominators

The metric version is `server_v2.2_strict_translation`: aligned scores use the canonical all-translation strict evaluator, with no finite-radius substitution or GT correction. Resource-limit/GT failures remain separately classified and cannot be replaced by approximations. `per_group.csv` contains all primary and secondary metric-specific valid row/building denominators and equal-building means, including the 12 description-provider × Builder × O/J combinations. P/E remain one observation per building and Builder.

`denominators.json` is reproduced verbatim below so planned, completed, valid, model failure, unknown, operationally missing, failure0-eligible and comparison denominators remain distinct.

```json
{json.dumps(results['denominators'], ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)}
```

## Inference

The unchanged `offline_evaluation/server_v2/analysis_v4.py::comparison_families()` defines 20 confirmatory contrasts in families 8/4/8. B−A signs are preserved. O/J pairs match within building and description provider; oracle contrasts average eligible O descriptions within building without replicating P/E. Common-valid differences are averaged within each building, and buildings receive equal weight. Independent N is at most60.

Seed {inference['seed']}; deterministic SHA256 counter RNG; building percentile bootstrap {inference['bootstrap_replicates']} replicates; two-sided building sign-flip {inference['signflip_replicates']} Monte Carlo replicates with plus-one correction, or the frozen exact sign enumeration when at most16 buildings. Holm adjustment is applied within each original family, retaining nonestimable comparisons in the planned multiplicity. `comparisons_20.csv` retains raw/adjusted p, CI, method, actual pair/building N, direction, and auxiliary bounded-IoU companion eligible N/effects. No supplementary significance tests are added by rendering.

`sensitivity_missing942.json` gives separate exploratory endpoint scenarios for hypothetical missing IoU in[0,1], with changes in pair/building availability. These are not observed zeros or confidence intervals; no sensitivity p-values are computed. MAE has no finite[0,1] bound. RQ2 status is {rq2['status']}; no unsupported matching or quality correlation is introduced.

## Tables and figures

Four LaTeX fragments use `booktabs` and `graphicx`: `table_main.tex`, `table_ablation.tex`, `table_failures.tex`, `table_statistics.tex`. NA means unavailable/inapplicable, never zero. All twenty contrasts are retained.

Figures are generated with matplotlib Agg as 300-dpi PNG and vector SVG. `figure_main` and `figure_description_builders` use exact `per_group` means and annotate valid slot counts; they have no inferred confidence intervals. Their units are separated into IoU and MAE panels. `figure_comparisons` uses all20 stored paired effects and bootstrap95% intervals, split by metric; labels include common-valid pair/building counts. The dashed zero reference is not a significance threshold. `figure_sources.json` identifies every plotted statistic and its JSON field; `render_artifacts.json` records output SHA256 and size. No figures use raw response contents or external images.

## CPU reproduction

Run from `/root/workspace/Minecraft-view-dataset` with the recorded code commit and isolated analysis Python. The integration command is:

```sh
{context.get('reproduction_command', 'UNSPECIFIED: integration must record the exact CPU-only command before publication.')}
```

Existing outputs are never overwritten. Rendering can also be replayed from the three stored JSON inputs into a fresh directory using `python -m offline_evaluation.analysis.goal007.render --results /absolute/results.json --context /absolute/render_context.json --rq2 /absolute/rq2_matched_scores_audit.json --output /absolute/new-output-directory`. Statistical reproduction additionally requires the immutable source audit gate and canonical slots; replaying a figure alone is not source revalidation.
"""
    return {'ARR_RESULTS_EN.md': en, 'SUMMARY_JA.md': ja, 'AUDIT.md': audit}


def _plot(results, output):
    # Lazy import keeps validation/table construction independent of plot setup.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'goal007-fixed60', 'axes.spines.top': False, 'axes.spines.right': False})
    lookup = _lookup(results)
    sources = []
    fixture_prefix = 'SYNTHETIC FIXTURE — NOT RESEARCH RESULTS\n' if results['input_kind'] == 'fixture' else ''

    def save(fig, name, provenance):
        for extension in ('png', 'svg'):
            with (output / f'{name}.{extension}').open('xb') as handle:
                fig.savefig(handle, format=extension, dpi=300, bbox_inches='tight',
                            metadata={'Software': 'Goal007 matplotlib renderer'} if extension == 'png' else {'Date': None, 'Creator': 'Goal007 matplotlib renderer'})
        plt.close(fig)
        csv_name = 'comparisons_20.csv' if name == 'figure_comparisons' else 'per_group.csv'
        sources.append({'figure': name, 'source': 'results.json',
                        'source_sha256': hashlib.sha256((output / 'results.json').read_bytes()).hexdigest(),
                        'source_csv': csv_name, 'source_csv_sha256': hashlib.sha256((output / csv_name).read_bytes()).hexdigest(),
                        'statistics': provenance,
                        'label': ANALYSIS_LABEL, 'recomputed_statistics': False})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    prov = []
    for ax, metric in zip(axes, METRICS):
        for offset, provider, color in [(-.16, 'openai', '#0072B2'), (.16, 'anthropic', '#D55E00')]:
            rows = [lookup[('builder_condition', provider, c, None, metric)] for c in CONDITIONS]
            xs = [i + offset for i in range(4)]
            for x, row in zip(xs, rows):
                value = row['equal_building_mean']
                if value is not None:
                    ax.plot(x, value, 'o', color=color, markersize=7)
                    ax.annotate(str(row['valid_rows']), (x, value), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8, color=color)
                prov.append({'per_group_key': [row[k] for k in ('group_level', 'provider', 'condition', 'description_model', 'metric')], 'field': 'equal_building_mean', 'annotation_field': 'valid_rows'})
            ax.plot([], [], 'o', color=color, label=NAMES[provider])
        ax.set_xticks(range(4), CONDITIONS)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(axis='y', alpha=.2)
        if metric == METRICS[0]: ax.set_ylim(0, 1.06)
        else: ax.set_ylim(bottom=0); ax.margins(y=.2)
        ax.set_title('Higher is better' if metric == METRICS[0] else 'Lower is better')
    axes[0].legend(frameon=False)
    fig.suptitle(fixture_prefix + 'Equal-building valid-only means\nLabels: valid slots; one operational observation remains missing', fontsize=11)
    save(fig, 'figure_main', prov)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout='constrained')
    pairs = [(b, d) for b in BUILDERS for d in ('openai', 'anthropic', 'gemini')]
    prov = []
    for ax, metric in zip(axes, METRICS):
        for offset, condition, color, marker in [(-.12, 'O', '#009E73', 'o'), (.12, 'J', '#CC79A7', 's')]:
            for i, (builder, description) in enumerate(pairs):
                row = lookup[('description_builder_condition', builder, condition, description, metric)]
                value = row['equal_building_mean']
                if value is not None:
                    ax.plot(i + offset, value, marker=marker, linestyle='none', color=color, markersize=6)
                    ax.annotate(str(row['valid_rows']), (i + offset, value), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7, color=color)
                prov.append({'per_group_key': [row[k] for k in ('group_level', 'provider', 'condition', 'description_model', 'metric')], 'field': 'equal_building_mean', 'annotation_field': 'valid_rows'})
            ax.plot([], [], marker=marker, linestyle='none', color=color, label=condition)
        ax.set_xticks(range(6), [f'{NAMES[d]}\n{NAMES[b]}' for b, d in pairs], fontsize=8)
        ax.set_xlabel('Description provider (top) / Builder (bottom)')
        ax.set_ylabel(METRIC_LABELS[metric]); ax.grid(axis='y', alpha=.2)
        if metric == METRICS[0]: ax.set_ylim(0, 1.06)
        else: ax.set_ylim(bottom=0); ax.margins(y=.2)
    axes[0].legend(frameon=False, title='Condition')
    fig.suptitle(fixture_prefix + 'Description provider × Builder: equal-building means\nLabels: valid slots; providers do not increase independent building N', fontsize=11)
    save(fig, 'figure_description_builders', prov)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), layout='constrained')
    prov = []
    palette = dict(zip(FAMILIES, ('#0072B2', '#009E73', '#D55E00')))
    for ax, metric in zip(axes, METRICS):
        rows = [r for r in results['comparisons'] if r['metric'] == metric]
        for i, row in enumerate(rows):
            y = len(rows) - 1 - i
            point, interval = row['mean_delta'], row['ci95']
            if point is not None:
                color = palette[row['family_id']]
                if interval is not None:
                    ax.plot(interval, [y, y], color=color, linewidth=1.6)
                ax.plot(point, y, 'o', color=color, markersize=5)
            else:
                ax.annotate('Not estimable', (0, y), fontsize=8)
            prov.append({'comparison_id': row['id'], 'fields': ['mean_delta', 'ci95', 'common_valid_pair_count', 'buildings'], 'family_id': row['family_id']})
        ax.set_yticks(range(len(rows)), [f"{_contrast(r)} ({r['common_valid_pair_count']}/{r['buildings']})" for r in reversed(rows)], fontsize=8)
        ax.axvline(0, color='black', linestyle='--', linewidth=.8)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('B - A: ' + METRIC_LABELS[metric])
        ax.set_title('Positive favors B' if metric == METRICS[0] else 'Positive means greater error for B')
        ax.set_ylim(-.7, len(rows) - .3)
    fig.suptitle(fixture_prefix + 'All predeclared paired effects with building bootstrap 95% intervals\nLabels: common-valid pairs / buildings; Holm p-values in the statistics table', fontsize=11)
    save(fig, 'figure_comparisons', prov)
    return sources


def render_results(results, output_dir, *, audit_context, rq2_audit):
    """Create all publication artifacts from statistics and explicit audit facts."""
    _validate(results, audit_context, rq2_audit)
    text_outputs = {'results.json': _json(results), 'per_group.csv': _csv(results['per_group']),
        'comparisons_20.csv': _csv(results['comparisons']), 'denominators.json': _json(results['denominators']),
        'sensitivity_missing942.json': _json(results['sensitivity_missing942']),
        'render_context.json': _json(audit_context), 'rq2_matched_scores_audit.json': _json(rq2_audit),
        'failure_taxonomy.csv': _csv(audit_context.get('failure_taxonomy', [])),
        'cost_verification.csv': _csv(audit_context.get('cost_rows', [])),
        'model_output_operational_errors.csv': _csv(audit_context.get('error_rows', [])),
        **_table_outputs(results), **_documents(results, audit_context, rq2_audit)}
    target = Path(output_dir)
    reserved = set(text_outputs) | {'figure_sources.json', 'render_artifacts.json'} | {name + '.' + ext for name in FIGURES for ext in ('png', 'svg')}
    if any((target / name).exists() or (target / name).is_symlink() for name in reserved):
        raise FileExistsError('render output exists; choose a fresh directory')
    # Keep matplotlib's font cache inside the explicitly requested output tree.
    target.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str((target / '.matplotlib-cache').resolve())
    # Fail before publication files when the isolated plotting dependency is absent.
    import matplotlib  # noqa: F401
    for name, value in text_outputs.items():
        with (target / name).open('x', encoding='utf-8', newline='') as handle:
            handle.write(value)
    sources = _plot(results, target)
    with (target / 'figure_sources.json').open('x', encoding='utf-8') as handle:
        handle.write(_json(sources))
    files = []
    for name in sorted(reserved - {'render_artifacts.json'}):
        path = target / name
        files.append({'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size})
    manifest = {'analysis_label': ANALYSIS_LABEL, 'input_kind': results['input_kind'], 'research_api_requests_sent': 0,
                'statistics_recomputed_by_renderer': False, 'matplotlib_version': matplotlib.__version__,
                'files': files, 'self_hash_excluded': True}
    with (target / 'render_artifacts.json').open('x', encoding='utf-8') as handle:
        handle.write(_json(manifest))
    return manifest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Bundle with results, audit_context and rq2_audit')
    parser.add_argument('--destination', '--output', dest='output', required=True, type=Path)
    for name in ('results', 'context', 'rq2'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.input:
        if any((args.results, args.context, args.rq2)):
            parser.error('--input cannot be combined with separate JSON inputs')
        bundle = json.loads(args.input.read_text())
        values, context, rq2 = bundle['results'], bundle['audit_context'], bundle['rq2_audit']
    else:
        if not all((args.results, args.context, args.rq2)):
            parser.error('provide --input or all of --results --context --rq2')
        values, context, rq2 = (json.loads(path.read_text()) for path in (args.results, args.context, args.rq2))
    result = render_results(values, args.output, audit_context=context, rq2_audit=rq2)
    print(json.dumps({'status': 'RENDERED', 'files': len(result['files']), 'research_api_requests_sent': 0}))


if __name__ == '__main__':
    main()
