"""CPU-only supplement identity, source retention and accounting checks."""
from copy import deepcopy
from pathlib import Path

import pytest

from offline_evaluation.operations.goal006b import planning
from offline_evaluation.continuation_v1 import runner as original_runner
from offline_evaluation.server_v2.generation import GenerationError, read_json, write_json, object_hash, utc_now


def rewrite(fixture, mutate):
    plan = read_json(fixture['path'])
    mutate(plan['identity'])
    plan['identity_hash'] = object_hash(plan['identity'])
    write_json(fixture['path'], plan)


def test_prepare_preserves_unknown_source_and_exact_twenty_payloads(fixture_plan):
    f = fixture_plan; plan = planning.load_plan(f['path']); identity = plan['identity']
    assert plan['api_allowed'] is False and plan['status'] == 'UNAPPROVED_SUPPLEMENT20'
    assert identity['schedule'] == f['source_plan']['identity']['schedule'][780:]
    assert [row['schedule_index'] for row in identity['schedule']] == list(range(940, 960))
    assert identity['intents'][0]['kind'] == 'manual_auth_retry'
    assert identity['intents'][0]['source_attempt_id'] == f['source_ledger']['entries'][-1]['attempt_id']
    assert all(row['kind'] == 'first_attempt' for row in identity['intents'][1:])
    assert identity['parent']['usage']['calls'] == 971 and identity['parent']['usage']['unknown_usage_attempts'] == 1
    assert identity['source_accounting']['unknown_usage_held_reservation'] == {
        'calls': 1, 'tokens': 432768, 'cost_usd_upper_bound': .165536}
    assert identity['budgets'] == {'max_new_calls': 20, 'max_total_calls': 991,
        'max_new_retries': 0, 'max_automatic_retries': 0, 'max_manual_auth_retries': 1,
        'max_total_retries': 5, 'max_attempts_per_input': 1, 'concurrency': 1,
        'max_tokens': 6655360, 'max_total_tokens': 268042881, 'max_cost_usd': 30.0,
        'max_output_tokens': 32768}
    for row in identity['schedule']:
        assert original_runner._payload(identity, row) == original_runner._payload(f['source_plan']['identity'], row)
    assert all(Path(path).read_bytes() == data for path, data in f['source_before'].items())
    assert not Path(identity['enabled_directory']).exists()
    assert not Path(identity['authority_path']).exists()


@pytest.mark.parametrize('mutation', [
    lambda i: i['schedule'].reverse(),
    lambda i: i['schedule'][0].update(input_text='changed'),
    lambda i: i['intents'][1].update(kind='manual_auth_retry'),
    lambda i: i['intents'][0].update(source_attempt_id='0' * 32),
    lambda i: i['budgets'].update(max_cost_usd=31),
    lambda i: i['budgets'].update(max_total_calls=990),
    lambda i: i['parent']['usage'].update(unknown_usage_attempts=0),
    lambda i: i['source_snapshot'].update(approval_sha256='wrong'),
    lambda i: i['source_snapshot'].update(files=[]),
    lambda i: i.update(prompt='changed'),
])
def test_rehashed_child_scope_tampering_is_rejected(fixture_plan, mutation):
    rewrite(fixture_plan, mutation)
    with pytest.raises(GenerationError):
        planning.load_plan(fixture_plan['path'])


def test_original_halt_controls_and_new_files_detected_without_replay(fixture_plan, monkeypatch):
    f = fixture_plan
    monkeypatch.setattr(original_runner, '_load_enabled', lambda *a: pytest.fail('historical781 validator replayed'))
    planning.load_plan(f['path'])
    source = f['source_directory'] / 'ledger.json'
    ledger = read_json(source); ledger['halt_reason'] = None; write_json(source, ledger)
    with pytest.raises(GenerationError, match='source_file_changed'):
        planning.load_plan(f['path'])
    source.write_bytes(f['source_before'][str(source)])
    (f['source_path'].parent / 'unexpected.json').write_text('{}')
    with pytest.raises(GenerationError, match='inventory_changed'):
        planning.load_plan(f['path'])


def test_copied_child_and_checkpoint_proof_tampering_rejected(fixture_plan):
    f = fixture_plan
    target = f['root'] / 'goal006b-copy'; target.mkdir()
    (target / 'plan.json').write_bytes(f['path'].read_bytes())
    with pytest.raises(GenerationError, match='canonical_scope'):
        planning.load_plan(target / 'plan.json')
    proof = read_json(f['proof_path']); proof['release_id'] += 1; write_json(f['proof_path'], proof)
    with pytest.raises(GenerationError, match='checkpoint_proof_changed'):
        planning.load_plan(f['path'])


def test_child_pricing_append_keeps_source_plan_approval_ledger_untouched(fixture_plan):
    f = fixture_plan; before = f['path'].read_bytes()
    document = deepcopy(f['plan']['identity']['pricing_document'])
    document['verified_at_utc'] = utc_now()
    source = f['path'].parent / 'today-official-fixture.json'; write_json(source, document)
    result = planning.refresh_pricing(f['path'], pricing_file=source)
    assert Path(result['revision_path']).parent == f['path'].parent / 'pricing-revisions'
    assert f['path'].read_bytes() == before
    assert not Path(f['plan']['identity']['enabled_directory']).exists()
    assert not Path(f['plan']['identity']['authority_path']).exists()
    assert all(Path(path).read_bytes() == data for path, data in f['source_before'].items())
    planning.current_pricing(f['plan']['identity'], fresh=True)
    document['openai']['input_usd_per_million'] += 1
    write_json(source, document)
    with pytest.raises(GenerationError, match='reapproval'):
        planning.refresh_pricing(f['path'], pricing_file=source)
