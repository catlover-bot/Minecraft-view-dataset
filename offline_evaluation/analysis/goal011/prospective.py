"""Unapproved design document from frozen evidence; no execution controls or IO."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import re

from ...resmoke_v2 import adapter
from ...server_v2.generation import object_hash
from .tradeoff import _gate, AuditFailure, digest

SAMPLING_FIELDS = ('temperature', 'top_p', 'seed', 'top_k', 'frequency_penalty', 'presence_penalty')
PLAN_LABELS = {'parent160', 'original781', 'supplement3', 'supplement17'}


def _require(condition, code):
    if not condition:
        raise AuditFailure(code)


def _frozen_protocol(package):
    plans = package['plans']
    _require(PLAN_LABELS <= set(plans), 'ALL_FOUR_FROZEN_PLANS_REQUIRED')
    sources = []; providers = {}; common = None; policies = []
    for label in sorted(PLAN_LABELS):
        plan = plans[label]; identity = plan['identity']
        _require(plan['identity_hash'] == object_hash(identity), 'FROZEN_PLAN_IDENTITY_HASH_MISMATCH')
        _require(hashlib.sha256(identity['prompt'].encode()).hexdigest() == identity['prompt_sha256'],
                 'FROZEN_PROMPT_HASH_MISMATCH')
        _require(object_hash(identity['schema']) == identity['schema_sha256'], 'FROZEN_SCHEMA_HASH_MISMATCH')
        cap = identity['budgets']['max_output_tokens']
        _require(type(cap) is int and cap == 32768, 'FROZEN_OUTPUT_CAP_CHANGED')
        protocol = (identity['prompt_sha256'], identity['schema_sha256'], cap)
        if common is None: common = protocol
        _require(common == protocol, 'FROZEN_PLANS_HAVE_DIFFERENT_PROTOCOLS')
        policy = identity.get('reservation_policy')
        if policy is not None:
            _require(isinstance(policy, str) and 'context' in policy and 'output' in policy,
                     'FROZEN_INPUT_RESERVATION_POLICY_UNSUPPORTED')
            policies.append({'source_label': label, 'plan_identity_hash': plan['identity_hash'], 'policy': policy})
        rows_by_provider = {}
        for row in identity['schedule']:
            provider = row['provider']
            _require(provider in ('openai', 'anthropic'), 'UNPLANNED_BUILDER_PROVIDER')
            if provider in rows_by_provider:
                _require(rows_by_provider[provider]['model'] == row['model'], 'WITHIN_PLAN_MODEL_DRIFT')
            else: rows_by_provider[provider] = row
        for provider, price in identity['pricing'].items():
            _require(provider in ('openai', 'anthropic'), 'UNPLANNED_PRICING_PROVIDER')
            context = price['context_window_tokens']
            _require(type(context) is int and context > 0, 'FROZEN_CONTEXT_RESERVATION_INVALID')
            model = price['model']
            record = rows_by_provider.get(provider)
            if record is None:
                continue
            _require(model == record['model'], 'FROZEN_PRICE_MODEL_BINDING_CHANGED')
            payload = adapter.build_payload(provider, model, identity['prompt'], record['input_text'], cap)
            schema = payload['text']['format']['schema'] if provider == 'openai' else payload['output_config']['format']['schema']
            _require(schema == identity['schema'], 'FROZEN_ADAPTER_SCHEMA_CHANGED')
            sampling = {field: {'explicit': field in payload, 'value': payload.get(field)} for field in SAMPLING_FIELDS}
            _require(not any(value['explicit'] for value in sampling.values()), 'FROZEN_SAMPLING_OMISSIONS_CHANGED')
            current = {'provider': provider, 'requested_model_id': model,
                       'historical_context_input_reservation_tokens': context, 'max_output_tokens': cap,
                       'sampling_fields': sampling, 'model_availability_now': 'NOT_QUERIED_NOT_VERIFIED',
                       'provider_defaults': 'NOT_CONTROLLED_BY_OMITTED_PAYLOAD_FIELDS',
                       'current_context_and_output_support': 'REQUIRES_FUTURE_OFFICIAL_VERIFICATION'}
            if provider in providers:
                _require(current == providers[provider], 'FROZEN_PROVIDER_PROTOCOL_DRIFT')
            providers[provider] = current
            sources.append({'source_label': label, 'plan_identity_hash': plan['identity_hash'],
                            'provider': provider, 'representative_payload_sha256': object_hash(payload),
                            'pricing_record_content_sha256': digest(price), 'payload_constructed_locally_only': True})
    _require(set(providers) == {'openai', 'anthropic'}, 'BOTH_BUILDER_PROTOCOLS_REQUIRED')
    _require({p['source_label'] for p in policies} >= {'parent160', 'original781'}, 'PARENT_RESERVATION_POLICY_REQUIRED')
    return {'prompt_sha256': common[0], 'schema_sha256': common[1], 'max_output_tokens': common[2],
            'providers': providers, 'source_plan_proofs': sources, 'historical_reservation_policies': policies,
            'reservation_interpretation': 'Historic full-context input plus output-cap reservation; not a measured token count, '
                'future context-fit proof, empirical completion cost, or invoice.'}


def _rest_evidence(rest_result):
    feasibility = rest_result['feasibility']; rows = rest_result['sources']
    selected = [r for r in rows if r.get('selected_for_prospective_design')]
    _require(len(selected) == 30 and len({r['scene_id'] for r in selected}) == 30,
             'EXACT_FROZEN_REST30_NO_REPLACEMENTS_REQUIRED')
    _require(all(r['description_provider'] == 'openai' for r in selected), 'PREFERRED_PROVIDER_CHANGED')
    _require(all(type(r['size_rank_stratum']) is int for r in selected)
             and Counter(r['size_rank_stratum'] for r in selected) == {q: 6 for q in range(5)}, 'REST30_FIVE_STRATA_CHANGED')
    _require(feasibility['independent_rest_buildings'] == 30 and feasibility.get('source_before_after_unchanged') is True,
             'REST_SOURCE_AUDIT_REQUIRED')
    _require(feasibility.get('rest_evaluation_overlap') == [] and feasibility.get('rest_smoke_overlap') == [],
             'REST_SPLIT_OVERLAP_OR_MISSING_PROOF')
    missing = [r['scene_id'] for r in selected if r.get('source_status') != 'VERIFIED_FROZEN_O_SOURCE']
    for row in selected:
        _require(all(re.fullmatch('[0-9a-f]{64}', str(row.get(k, ''))) for k in
                     ('gt_sha256', 'bbox_sha256', 'scene_source_manifest_sha256')), 'REST_GT_SOURCE_HASH_MISSING')
        if row['scene_id'] not in missing:
            _require(all(re.fullmatch('[0-9a-f]{64}', str(row.get(k, ''))) for k in ('input_sha256', 'typed_tree_sha256')),
                     'REST_INPUT_HASH_MISSING')
    _require(feasibility.get('verified_selected_inputs') == 30-len(missing), 'REST_INPUT_AVAILABILITY_COUNT_MISMATCH')
    keep = ('scene_id', 'size_rank_stratum', 'description_provider', 'source_status', 'input_sha256',
            'typed_tree_sha256', 'gt_sha256', 'bbox_sha256', 'scene_source_manifest_sha256',
            'input_utf8_bytes', 'input_characters', 'future_context_fit_status', 'future_output_length_status',
            'description_request_sha256', 'request_view_count_status', 'actual_description_request_view_count',
            'prior_use_status', 'exact_v4_prior_builder_artifact_references', 'truly_unseen_anywhere')
    return {'feasibility_sha256': digest(feasibility), 'rest_source_rows_sha256': digest(rows),
            'source_inventory_sha256': feasibility.get('source_inventory_sha256'),
            'selected_inputs': [{k: r.get(k) for k in keep} for r in sorted(selected, key=lambda x: x['scene_id'])],
            'missing_selected_input_scenes': sorted(missing), 'verified_inputs': 30-len(missing),
            'not_evaluated_in_verified_fixed_plans': feasibility.get('not_evaluated_in_verified_fixed_plans'),
            'prior_exposure_audit_sha256': digest(feasibility.get('prior_exposure_audit', {})),
            'truly_unseen_anywhere': 'NOT_VERIFIED', 'new_distribution': False,
            'readiness_status': 'NOT_VERIFIED_NO_PAID_EXECUTION_APPROVAL',
            'no_silent_scene_or_provider_substitution': True}


def design_protocol(package, tradeoff_result, rest_result, gate):
    """Return {protocol, report_markdown}; never create a plan, authority or launcher."""
    _gate(gate, gate.get('plan_sha256'))
    before = digest([package, tradeoff_result, rest_result, gate])
    _require(digest(package) == gate['source_package_sha256'], 'SOURCE_PACKAGE_GATE_MISMATCH')
    _require(tradeoff_result['claim_audit']['status'] == 'PASS', 'PASSING_TRADEOFF_AUDIT_REQUIRED')
    _require(tradeoff_result['metadata']['gate'] == gate, 'TRADEOFF_GATE_MISMATCH')
    rest_gate = rest_result['feasibility'].get('gate', {})
    _require(all(rest_gate.get(k) == gate[k] for k in ('input_kind', 'question_register_sha256', 'code_commit'))
             and rest_gate.get('analysis_plan_sha256', rest_gate.get('plan_sha256')) == gate['plan_sha256'],
             'REST_GATE_MISMATCH')
    saved_precision = tradeoff_result['precision_design_inputs']
    _require(len(saved_precision) == 12, 'ALL_PROVIDER_PRECISION_DESIGNS_REQUIRED')
    precision = [r for r in saved_precision if r['description_provider'] == 'openai']
    _require({(r['builder'], r['metric_id']) for r in precision}
             == {(b, m) for b in ('openai', 'anthropic') for m in ('occupancy_iou', 'surface_f1_at_1')}
             and len(precision) == 4, 'FOUR_PREFERRED_PROVIDER_PRECISION_INPUTS_REQUIRED')
    historical = _frozen_protocol(package); rest = _rest_evidence(rest_result)
    candidates = []
    for repeats in (1, 3):
        by_provider = []
        for provider, source in sorted(historical['providers'].items()):
            calls = 30*repeats
            input_tokens = calls*source['historical_context_input_reservation_tokens']
            output_tokens = calls*source['max_output_tokens']
            by_provider.append({'provider': provider, 'requested_model_id': source['requested_model_id'],
                'planned_calls': calls, 'input_reservation_tokens': input_tokens, 'output_reservation_tokens': output_tokens,
                'total_reserved_tokens': input_tokens+output_tokens,
                'symbolic_cost_usd': f'({input_tokens} * future_{provider}_input_USD_per_1M + '
                    f'{output_tokens} * future_{provider}_output_USD_per_1M) / 1000000',
                'official_rate_or_billable_feature_verification': 'REQUIRES_CURRENT_PRICE_VERIFICATION',
                'numeric_cost_usd': None})
        candidates.append({'candidate_id': 'R'+str(repeats), 'status': 'UNAPPROVED_NOT_RUN_NOT_ENABLED',
            'condition': 'O', 'description_provider': 'openai', 'planned_buildings': 30,
            'builders': 2, 'repeats_per_building_builder': repeats, 'planned_calls': 60*repeats,
            'total_input_reservation_tokens': sum(r['input_reservation_tokens'] for r in by_provider),
            'total_output_reservation_tokens': sum(r['output_reservation_tokens'] for r in by_provider),
            'total_reserved_tokens': sum(r['total_reserved_tokens'] for r in by_provider),
            'per_provider': by_provider, 'independent_N': 30, 'retries': 0,
            'pricing_status': 'REQUIRES_CURRENT_PRICE_VERIFICATION', 'numeric_cost_usd': None,
            'precision_scenarios': [{'builder': r['builder'], 'metric_id': r['metric_id'],
                'observed_valid_buildings': r['observed_valid_buildings'],
                'eval60_delta_sd': r['building_delta_sample_sd_ddof1'],
                'building_delta_vector_sha256': r['building_delta_vector_sha256'],
                'scenarios': copy.deepcopy([s for s in r['designs'] if s['repeat_generations_per_building'] == repeats]),
                'limitations': r['limitations']} for r in precision]})
    protocol = {'status': 'UNAPPROVED_NOT_RUN_NO_LAUNCHER', 'analysis_label': 'PROSPECTIVE_UNAPPROVED_DESIGN_INFORMED_BY_EVAL60',
        'registration_claim': False, 'gate': dict(gate), 'historical_protocol': historical, 'rest_evidence': rest,
        'primary_estimand': 'equal-building mean aligned occupancy IoU: Builder minus same-input deterministic filled-box',
        'critical_counter_metrics': ['surface_f1_at_1', 'occupancy_precision', 'occupancy_recall',
                                    'known_model_output_failure_rate', 'operational_missing_rate'],
        'description_provider': 'openai', 'provider_selection': 'Fixed in QUESTION_REGISTER/ANALYSIS_PLAN before new Goal011 arithmetic; '
            'historical provenance and shared 1800-cap description design, not favorable downstream effect selection.',
        'candidates': candidates, 'conditional_candidate_preference': 'R1_IF_FUTURE_INDEPENDENT_VALIDATION_IS_SEPARATELY_JUSTIFIED',
        'preference_reason': 'R1 uses 60 calls and R3 uses 180; single-generation eval60 cannot identify within-building repeat variance '
            'or establish the precision gain needed to justify R3. Neither 30 buildings nor either configuration is certified sufficient.',
        'one_builder_scope_limit': 'Dropping one Builder reduces calls but cannot independently check the trade-off for both systems; '
            'not added as a third candidate.',
        'validity_and_missing_policy': {'primary': 'common-valid identical core-mask; average eligible repeats within each building before equal-building mean',
            'secondary': 'separate bounded known-definitive-output-failure0 sensitivity and independent failure/unknown rates',
            'unknown_or_resource_missing': 'NULL, never0; retain fixed planned buildings, attempt identities and missing reasons',
            'successful_empty_output': 'Retain as normal success; undefined metrics remain NULL and never reclassify as model failure.',
            'no_eligible_repeat_for_building': 'Exclude only from the conditional common-valid mean; retain planned N30 and report actual eligible buildings and pairs for each Builder.',
            'reliability_denominator': 'All fixed planned slots per Builder:30*r; unknown/operational missing and known output failures remain separate.',
            'repeated_samples_are_not_independent_buildings': True},
        'baseline': 'Same frozen O dimensions_estimate rounding, default world origin0 and fixed filled-box, no GT in construction; '
            'GT strict occupancy translation reused only for evaluation.',
        'future_checks_before_approval': ['verify selected rest input availability without silent replacement',
            'resolve previous-exposure scope and acceptable independent-validation claim',
            'verify current exact model IDs, schema support, context/output limits and official prices on execution date',
            'freeze a separate future protocol before inspecting its outcomes',
            'obtain explicit new cost/call/token scope and fresh real-TTY human authorization'],
        'unknown_usage_policy': 'Reserve before any future POST; retain unknown-use holds, halt on uncertain response, no automatic resend or retry.',
        'checkpoint_policy': 'Only a separately approved future plan may create fresh local immutable checkpoints and a new GitHub draft; '
            'verify byte digests and restoration, halt on save failure, never modify the existing three drafts.',
        'pricing': {'status': 'REQUIRES_CURRENT_PRICE_VERIFICATION', 'historical_dollar_prices_reused': False,
                    'empirical_cost_prediction_usd': None, 'invoice_usd': None,
                    'symbolic_formula_requires_verification_of_all_billable_features': True},
        'causal_limitation': 'Repeating O on the same saved JSON cannot identify information lost in image-to-text conversion. '
            'That would require a separate controlled paired intervention on available spatial information/input modality, '
            'holding tasks and relevant generation settings fixed; it is not implemented here.',
        'historical_991_calls_or_USD30_are_not_authorization': True,
        'research_api_requests_sent': 0, 'model_queries': 0, 'token_count_api_requests': 0, 'api_key_queries': 0,
        'new_authority_ledger_approval_or_launcher_created': False, 'execution_command': None,
        'source_bundle_before_after_unchanged': True}
    _require(digest([package, tradeoff_result, rest_result, gate]) == before, 'PROSPECTIVE_INPUT_MUTATION')
    return {'protocol': protocol, 'report_markdown': render_protocol(protocol)}


def render_protocol(protocol):
    lines = ['# rest30の将来検証案：未承認・未実行', '',
        '**UNAPPROVED_NOT_RUN_NO_LAUNCHER**。60建築の既存結果を見た上で作成した設計案であり、事前登録・生成承認ではない。', '',
        '問いは同じO説明を使うBuilderと固定filled-boxの指標トレードオフに限定する。主estimandは建築等重みのaligned occupancy IoU差（Builder−box）。surface F1@1を必須counter-metricとし、Precision・Recall・出力失敗率・運用欠測率も併記する。', '',
        '説明元は計画でOpenAIに固定済みで、従来のprovenanceと1800-capの説明生成設計を理由とする。既存の効果が有利なproviderを計算後に選ぶ規則ではない。入力が不足しても建築やproviderを黙って差し替えない。', '',
        '| 候補 | 建築N | Builders | r | 将来calls | 入力予約tokens | 出力予約tokens | 予約総tokens |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in protocol['candidates']:
        lines.append(f"| {c['candidate_id']} | 30 | 2 | {c['repeats_per_building_builder']} | {c['planned_calls']} | "
            f"{c['total_input_reservation_tokens']} | {c['total_output_reservation_tokens']} | {c['total_reserved_tokens']} |")
    lines += ['', '各候補の予約額は、次のBuilder別の式を足し合わせる。単価はすべて将来実行日の公式USD/1M tokensで、現在は未確認。追加課金項目や料金tierがあれば承認前に式へ反映する。', '',
              '| 候補 | Builder | USD予約額の記号式 |', '| --- | --- | --- |']
    for candidate in protocol['candidates']:
        for provider in candidate['per_provider']:
            lines.append(f"| {candidate['candidate_id']} | {provider['provider']} | `{provider['symbolic_cost_usd']}` |")
    lines += ['', '予約は旧planのfull-context input＋output capに基づく保守的上限で、実入力token数・完了予測・将来のcontext適合証明ではない。料金は将来の公式単価と課金対象機能の確認が必要（REQUIRES_CURRENT_PRICE_VERIFICATION）。過去価格から新たなドル値を作らず、請求書や消費実績と区別する。', '',
              '下式のfuture単価は将来実行日の公式USD/100万tokensを表す未確定変数で、値を代入していない。各候補の合計予約費用は2 providerの式の和である。別の課金対象機能があればその適用条件も別途検証する。', '',
              '| 候補 | Builder | 料金予約の記号式（USD、未確定） |',
              '| --- | --- | --- |']
    for c in protocol['candidates']:
        for source in c['per_provider']:
            lines.append(f"| {c['candidate_id']} | {source['provider']} | `{source['symbolic_cost_usd']}` |")
    lines += ['',
              '| Builder | 保存requested model ID | 入力予約/回 | 出力上限/回 | sampling |',
              '| --- | --- | --- | --- | --- |']
    for p in protocol['historical_protocol']['providers'].values():
        lines.append(f"| {p['provider']} | {p['requested_model_id']} | {p['historical_context_input_reservation_tokens']} | "
                     f"{p['max_output_tokens']} | seed/temperature/top-p等はpayloadで省略 |")
    lines += ['', 'モデルの現在の利用可能性は照会していない。省略されたsamplingパラメータの既定値や決定性は保証しない。prompt SHA：`'+protocol['historical_protocol']['prompt_sha256']+'`、schema SHA：`'+protocol['historical_protocol']['schema_sha256']+'`。保存planから同一性を検査した。', '',
              '| Builder | 指標 | eval60 valid N | SD | 将来N | r | t近似半幅の仮定範囲 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for c in protocol['candidates']:
        for p in c['precision_scenarios']:
            sd = p['eval60_delta_sd']; sdtext = 'NULL' if sd is None else f'{sd:.6f}'
            for s in p['scenarios']:
                width = s['approximate_halfwidth_range']
                text = 'NULL' if width is None else f'[{width[0]:.6f}, {width[1]:.6f}]'
                lines.append(f"| {p['builder']} | {p['metric_id']} | {p['observed_valid_buildings']} | {sdtext} | "
                             f"{s['target_buildings']} | {s['repeat_generations_per_building']} | {text} |")
    lines += ['', '表のN60は精度を比較するための仮想参照であり、第3の実験候補ではない。元restは30建築で、N60への拡張可能性や追加コホートの利用は確認していない。', '',
        '半幅はt(0.975,N−1)×eval60の建築差SD/√Nの計画近似である。0.03は許容幅として置いた値で、検証済みMCIDではない。r3の範囲は非負の建築間/反復内分散分解と条件付き独立反復を仮定し、r1半幅の1/√3〜1倍。単発データから反復内分散は識別できず、相関反復・valid選択・コホート差があれば範囲は保証されない。N30の十分性や検出力を主張しない。', '',
        '将来検証を別途正当化する場合の第一候補はR1とする。R3の追加120 callsを支持する反復分散の証拠が現在はなく、建築Nも30のままである。これは実行の推奨確定・承認ではない。1 Builderへ減らせば他方の独立確認を失うため、第3候補は追加しない。', '',
        f"rest30の選択済み入力は{protocol['rest_evidence']['verified_inputs']}/30件がsource auditで確認済み。過去利用の証拠と未確認範囲はREST30_FEASIBILITY/SOURCESに残す。Goal005–010で未評価であっても完全未見・新分布とは呼ばない。", '',
        '将来はcommon-validの同一core集合と建築内の反復平均を先に固定し、既知出力失敗のbounded failure0は別表にする。HTTP520・unknown usage・resource missingはNULLのまま維持する。従来991 callsとUSD30枠は新承認ではない。新scope・現行価格/モデル/上限の公式確認・新しい実TTY承認が別途必要で、自動retryは0。保存失敗や未知応答では停止し、過去Draftを変更しない。本Goalは台帳・authority・approval・launcherを作らず、実行コマンドも提示しない。', '',
        '正常な空出力は成功分類を維持し、未定義metricを0や失敗へ置き換えない。適格反復がない建築は条件付きcommon-valid平均からのみ除外し、予定N30と実際の適格建築数・pair数をBuilder別に併記する。出力失敗率と運用欠測率は各Builderの固定30×r slotsを分母に分けて報告する。', '',
        '同じO JSONから何回生成しても、image→textで失われた情報の因果効果は識別できない。入力情報への制御されたpaired介入を含む別研究が必要で、ここでは実装しない。研究API・モデル・token-count API・キー照会は0。', '']
    return '\n'.join(lines)
