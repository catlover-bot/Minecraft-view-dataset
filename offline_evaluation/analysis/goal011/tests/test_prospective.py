import copy
import hashlib
import json

import pytest

from offline_evaluation.analysis.goal011 import prospective as p
from offline_evaluation.analysis.goal011 import tradeoff
from offline_evaluation.resmoke_v2.schema import action_schema
from offline_evaluation.server_v2.generation import object_hash


def inputs():
    prompt = 'fixture-only frozen prompt'
    identity = {'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                'schema': action_schema(), 'schema_sha256': object_hash(action_schema()),
                'budgets': {'max_output_tokens': 32768},
                'pricing': {provider: {'model': model, 'context_window_tokens': context,
                                      'old_input_USD_per_1M': 1234567}
                            for provider, model, context in [('openai', 'gpt-5-mini', 400000),
                                ('anthropic', 'claude-haiku-4-5-20251001', 200000)]},
                'schedule': [{'provider': provider, 'model': model, 'input_text': '{}'}
                             for provider, model in p.adapter.MODELS.items()]}
    plans = {}
    for label in p.PLAN_LABELS:
        i = copy.deepcopy(identity)
        if label in ('parent160', 'original781'):
            i['reservation_policy'] = 'full documented context input plus output cap; not completion cost'
        plans[label] = {'identity': i, 'identity_hash': object_hash(i)}
    package = {'plans': plans}
    gate = {'status': 'PASS', 'input_kind': 'fixture', 'plan_sha256': 'a'*64,
            'question_register_sha256': 'b'*64, 'source_package_sha256': tradeoff.digest(package),
            'code_commit': 'c'*40, 'specification_commit': 'd'*40}
    building = [{'description_provider': provider, 'builder': builder, 'metric_id': metric,
                 'structure': 'filled_box', 'frame': 'aligned', 'scene_id': str(i), 'delta': value}
                for provider in tradeoff.PROVIDERS for builder in tradeoff.BUILDERS
                for metric in ('occupancy_iou', 'surface_f1_at_1') for i, value in enumerate([-.1, .1, .3])]
    result = {'claim_audit': {'status': 'PASS'}, 'metadata': {'gate': copy.deepcopy(gate)},
              'precision_design_inputs': tradeoff.precision_design_inputs(building)}
    rest = {'feasibility': {'independent_rest_buildings': 30, 'source_before_after_unchanged': True,
                           'gate': {**copy.deepcopy(gate), 'analysis_plan_sha256': gate['plan_sha256']},
                           'rest_evaluation_overlap': [], 'rest_smoke_overlap': [], 'verified_selected_inputs': 30,
                           'not_evaluated_in_verified_fixed_plans': True, 'source_inventory_sha256': '9'*64,
                           'prior_exposure_audit': {}},
            'sources': [{'scene_id': f'rest{i:02}', 'size_rank_stratum': i//6, 'description_provider': 'openai',
                         'selected_for_prospective_design': True, 'source_status': 'VERIFIED_FROZEN_O_SOURCE',
                         'input_sha256': str(i%10)*64, 'typed_tree_sha256': '8'*64, 'gt_sha256': '7'*64,
                         'bbox_sha256': '6'*64, 'scene_source_manifest_sha256': '5'*64,
                         'input_utf8_bytes': 100, 'input_characters': 100,
                         'future_context_fit_status': 'NOT_VERIFIED', 'future_output_length_status': 'NOT_VERIFIED',
                         'prior_use_status': 'EXPOSURE_REFERENCES_FOUND', 'exact_v4_prior_builder_artifact_references': 0,
                         'truly_unseen_anywhere': 'NOT_VERIFIED'} for i in range(30)]}
    return package, result, rest, gate


def rebind(args):
    package, result, rest, gate = args
    for plan in package['plans'].values():plan['identity_hash'] = object_hash(plan['identity'])
    gate['source_package_sha256'] = tradeoff.digest(package)
    result['metadata']['gate'] = copy.deepcopy(gate)
    rest['feasibility']['gate'] = {**copy.deepcopy(gate), 'analysis_plan_sha256': gate['plan_sha256']}
    return args


def test_two_unapproved_candidates_symbolic_cost_and_immutability():
    args = inputs(); before = copy.deepcopy(args); result = p.design_protocol(*args)
    assert args == before
    protocol = result['protocol']; a, b = protocol['candidates']
    assert protocol['status'] == 'UNAPPROVED_NOT_RUN_NO_LAUNCHER'
    assert (a['planned_calls'], b['planned_calls']) == (60, 180)
    assert (a['total_input_reservation_tokens'], a['total_output_reservation_tokens']) == (18000000, 1966080)
    assert (b['total_input_reservation_tokens'], b['total_output_reservation_tokens']) == (54000000, 5898240)
    assert a['independent_N'] == b['independent_N'] == 30
    assert all(c['numeric_cost_usd'] is None and c['retries'] == 0 for c in (a, b))
    assert '1234567' not in json.dumps(result)  # obsolete prices not copied or used
    assert 'fixture-only frozen prompt' not in json.dumps(result)  # no raw prompt
    assert protocol['research_api_requests_sent'] == 0 and protocol['execution_command'] is None
    assert protocol['new_authority_ledger_approval_or_launcher_created'] is False
    assert protocol['pricing']['invoice_usd'] is None
    assert all(len(c['precision_scenarios']) == 4 for c in (a,b))
    assert all(provider['symbolic_cost_usd'] in result['report_markdown'] for c in (a,b) for provider in c['per_provider'])
    assert len(protocol['rest_evidence']['selected_inputs']) == 30
    json.dumps(result, allow_nan=False)


def test_gate_precedes_package_access():
    args = inputs(); args[3]['status'] = 'FAIL'
    with pytest.raises(tradeoff.AuditFailure):p.design_protocol(None, None, None, args[3])


@pytest.mark.parametrize('change', ['source', 'plan_identity', 'prompt', 'schema', 'cap', 'model', 'context', 'policy', 'missing_plan'])
def test_frozen_protocol_drift_rejected(change):
    args = list(inputs()); package = args[0]; plan = package['plans']['original781']; identity = plan['identity']
    if change == 'source':package['extra'] = True
    elif change == 'plan_identity':plan['identity_hash'] = '0'*64
    elif change == 'prompt':identity['prompt'] += ' changed'
    elif change == 'schema':identity['schema']['extra'] = True
    elif change == 'cap':identity['budgets']['max_output_tokens'] = 100
    elif change == 'model':identity['schedule'][0]['model'] = 'new-snapshot'
    elif change == 'context':identity['pricing']['openai']['context_window_tokens'] = 1
    elif change == 'policy':del identity['reservation_policy']
    elif change == 'missing_plan':del package['plans']['supplement3']
    if change not in ('source','plan_identity'):args = rebind(args)
    elif change == 'plan_identity':
        args[3]['source_package_sha256'] = tradeoff.digest(package); args[1]['metadata']['gate'] = copy.deepcopy(args[3])
    with pytest.raises((tradeoff.AuditFailure, p.adapter.ProviderError)):p.design_protocol(*args)


def test_tradeoff_gate_and_saved_design_must_match():
    args = inputs(); args[1]['metadata']['gate']['code_commit'] = '0'*40
    with pytest.raises(tradeoff.AuditFailure, match='TRADEOFF_GATE_MISMATCH'):p.design_protocol(*args)
    args = inputs(); args[1]['precision_design_inputs'].pop()
    with pytest.raises(tradeoff.AuditFailure, match='ALL_PROVIDER_PRECISION_DESIGNS_REQUIRED'):p.design_protocol(*args)


def test_rest_gate_must_bind_same_frozen_plan_and_code():
    args = inputs(); args[2]['feasibility']['gate']['analysis_plan_sha256'] = '0'*64
    with pytest.raises(tradeoff.AuditFailure, match='REST_GATE_MISMATCH'):p.design_protocol(*args)


def test_missing_preferred_source_retained_without_switch_or_enable():
    args = inputs(); rest = args[2]
    rest['sources'][0].update(source_status='NOT_AVAILABLE', input_sha256=None, typed_tree_sha256=None)
    rest['feasibility']['verified_selected_inputs'] = 29
    r = p.design_protocol(*args)['protocol']
    assert r['rest_evidence']['verified_inputs'] == 29
    assert r['rest_evidence']['missing_selected_input_scenes'] == ['rest00']
    assert r['rest_evidence']['readiness_status'] == 'NOT_VERIFIED_NO_PAID_EXECUTION_APPROVAL'
    assert r['candidates'][0]['planned_buildings'] == 30 and r['candidates'][0]['planned_calls'] == 60


@pytest.mark.parametrize('change', ['duplicate', 'provider', 'stratum', 'overlap', 'availability', 'hash'])
def test_rest_identity_or_count_mismatch_rejected(change):
    args = inputs(); rest = args[2]
    if change == 'duplicate':rest['sources'][0]['scene_id'] = rest['sources'][1]['scene_id']
    if change == 'provider':rest['sources'][0]['description_provider'] = 'gemini'
    if change == 'stratum':rest['sources'][0]['size_rank_stratum'] = 1
    if change == 'overlap':rest['feasibility']['rest_evaluation_overlap'] = ['rest00']
    if change == 'availability':rest['feasibility']['verified_selected_inputs'] = 29
    if change == 'hash':rest['sources'][0]['input_sha256'] = None
    with pytest.raises(tradeoff.AuditFailure):p.design_protocol(*args)


def test_sampling_omission_is_measured_from_safe_payload_construction(monkeypatch):
    args = inputs(); original = p.adapter.build_payload
    def changed(*a, **kw):
        result = original(*a, **kw); result['temperature'] = 0
        return result
    monkeypatch.setattr(p.adapter, 'build_payload', changed)
    with pytest.raises(tradeoff.AuditFailure, match='FROZEN_SAMPLING_OMISSIONS_CHANGED'):p.design_protocol(*args)


def test_no_network_keys_or_file_access(monkeypatch):
    import builtins
    import os
    import socket
    args = inputs()
    def blocked(*a, **kw):raise AssertionError('forbidden IO')
    monkeypatch.setattr(builtins, 'open', blocked)
    monkeypatch.setattr(socket, 'socket', blocked)
    monkeypatch.setattr(os, 'getenv', blocked)
    monkeypatch.setattr(p.adapter, 'send_request', blocked)
    monkeypatch.setattr(p.adapter.providers, 'send_request', blocked)
    result = p.design_protocol(*args)
    assert result['protocol']['research_api_requests_sent'] == 0


@pytest.mark.parametrize('missing_field', ['rest_evaluation_overlap', 'rest_smoke_overlap'])
def test_missing_overlap_proof_does_not_become_verified_disjointness(missing_field):
    args = inputs(); del args[2]['feasibility'][missing_field]
    with pytest.raises(tradeoff.AuditFailure, match='REST_SPLIT_OVERLAP_OR_MISSING_PROOF'): p.design_protocol(*args)


def test_empty_outputs_missing_buildings_and_reliability_denominators_remain_explicit():
    protocol = p.design_protocol(*inputs())['protocol']
    policy = protocol['validity_and_missing_policy']
    assert 'normal success' in policy['successful_empty_output']
    assert 'NULL' in policy['successful_empty_output']
    assert 'conditional common-valid mean' in policy['no_eligible_repeat_for_building']
    assert '30*r' in policy['reliability_denominator']
