"""Offline scope, accounting, immutable ancestors and pricing evidence checks."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from offline_evaluation.operations.goal006c import planning, runner
from offline_evaluation.server_v2.generation import GenerationError, read_json, write_json, object_hash, json_bytes, utc_now


def rewrite(fixture, edit):
    plan = read_json(fixture['path']); edit(plan['identity']); plan['identity_hash'] = object_hash(plan['identity'])
    write_json(fixture['path'], plan)


def test_exact_unattempted_seventeen_payload_bytes_and_two_holds(fixture_plan):
    plan = planning.load_plan(fixture_plan['path']); identity = plan['identity']; source = fixture_plan['source_plan']['identity']
    assert [r['schedule_index'] for r in identity['schedule']] == list(range(943, 960))
    assert len({(r['provider'], r['input_id']) for r in identity['schedule']}) == 17
    assert all(i['kind']=='first_attempt' and i['source_attempt_id'] is None for i in identity['intents'])
    for new, old in zip(identity['schedule'], source['schedule'][3:]):
        assert json_bytes(planning.base._payload(identity,new)) == json_bytes(planning.base._payload(source,old))
    assert identity['parent']['usage']['calls']==974 and identity['parent']['usage']['retries']==5
    assert identity['parent']['usage']['unknown_usage_attempts']==2
    holds=identity['source_accounting']['unknown_usage_holds']
    assert [h['cost_usd_upper_bound'] for h in holds]==[0.165536,0.165536]
    assert identity['budgets']['max_new_calls']==17 and identity['budgets']['max_total_calls']==991
    assert identity['budgets']['max_cost_usd']==30 and identity['budgets']['max_manual_auth_retries']==0
    assert identity['budgets']['max_automatic_retries']==0 and identity['budgets']['max_output_tokens']==32768
    assert plan['api_allowed'] is False and identity['new_scope_approved'] is False
    assert not Path(identity['enabled_directory']).exists() and not Path(identity['authority_path']).exists()
    assert all(Path(p).read_bytes()==b for p,b in fixture_plan['source_before'].items())


@pytest.mark.parametrize('change', ['include_520','include_940','reorder','input','prompt','model','cap','gt','budget','retry','parent_cost','unknown_hold','checkpoint','source_inventory','original_snapshot'])
def test_rehashed_mutation_cannot_expand_scope(fixture_plan, change):
    def edit(i):
        if change=='include_520':i['schedule'][0]=deepcopy(fixture_plan['source_plan']['identity']['schedule'][2])
        elif change=='include_940':i['schedule'][0]=deepcopy(fixture_plan['source_plan']['identity']['schedule'][0])
        elif change=='reorder':i['schedule'][0],i['schedule'][1]=i['schedule'][1],i['schedule'][0]
        elif change=='input':i['schedule'][0]['input_text']='different'
        elif change=='prompt':i['prompt']='different'
        elif change=='model':i['schedule'][0]['model']='replacement-model'
        elif change=='cap':i['budgets']['max_output_tokens']=32769
        elif change=='gt':i['gt_by_scene']={}
        elif change=='budget':i['budgets']['max_cost_usd']=31
        elif change=='retry':i['intents'][0]['kind']='manual_auth_retry'
        elif change=='parent_cost':i['parent']['usage']['cost_usd_upper_bound']=0
        elif change=='unknown_hold':i['source_accounting']['unknown_usage_holds'][1]['cost_usd_upper_bound']=0
        elif change=='checkpoint':i['checkpoint_reference']['archive_sha256']='0'*64
        elif change=='source_inventory':i['source_snapshot']['files'].pop()
        elif change=='original_snapshot':i['original_source_snapshot']['ledger_sha256']='0'*64
    rewrite(fixture_plan,edit)
    with pytest.raises(GenerationError):planning.load_plan(fixture_plan['path'])


def test_copied_plan_cannot_activate(fixture_plan):
    path=fixture_plan['root']/'goal006c-copy'/'plan.json';path.parent.mkdir();shutil.copyfile(fixture_plan['path'],path)
    with pytest.raises(GenerationError,match='canonical'):planning.load_plan(path)


def test_mutated_source_bytes_rejected(fixture_plan):
    path=Path(fixture_plan['source_path']);path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(GenerationError):planning.load_plan(fixture_plan['path'])


def test_missing_github_asset_proof_rejected(fixture_plan):
    proof=read_json(fixture_plan['proof_path']);proof['assets'].pop();write_json(fixture_plan['proof_path'],proof)
    with pytest.raises(GenerationError):planning.load_plan(fixture_plan['path'])


def test_current_pricing_append_is_C_only_preserves_old_bytes(fixture_plan):
    original=read_json(fixture_plan['path']);source_before=fixture_plan['source_before']
    result=planning.refresh_pricing(fixture_plan['path'],pricing_file=fixture_plan['pricing_path'])
    assert result['both_predecessors_unchanged'] is True
    assert read_json(fixture_plan['path'])==original
    assert all(Path(p).read_bytes()==b for p,b in source_before.items())
    assert planning.current_pricing(original['identity'],fresh=True)
    revision=read_json(Path(result['revision_path']));assert revision['previous_hash']==original['identity']['pricing_document_hash']


@pytest.mark.parametrize('change',['price','model','output_cap','future','stale'])
def test_price_refresh_rejects_changed_terms_or_bad_dates(fixture_plan, change):
    path=fixture_plan['pricing_path'];doc=read_json(path)
    if change=='price':doc['openai']['input_usd_per_million']=0.26
    elif change=='model':doc['openai']['model']='replacement-model'
    elif change=='output_cap':doc['openai']['reservation_output_tokens']=32769
    else:
        day='2099-01-01' if change=='future' else '2020-01-01'
        doc['verified_on']=day;doc['verified_at_utc']=day+'T00:00:00Z'
        for provider in ('openai','anthropic'):doc[provider]['verified_on']=day
    write_json(path,doc)
    with pytest.raises(GenerationError):planning.refresh_pricing(fixture_plan['path'],pricing_file=path)
    assert not (fixture_plan['path'].parent/'pricing-revisions').exists()


def test_cumulative_cost_limit_includes_existing_cost_and_both_holds(fixture_plan):
    source=deepcopy(fixture_plan['source_plan'])
    source['identity']['parent']['usage']['cost_usd_upper_bound']=29.0
    with pytest.raises(GenerationError,match='cumulative_cost_limit'):
        planning.budgets(source,fixture_plan['source_ledger'])


def test_explicit_commit_must_exist_and_bind_current_code():
    with pytest.raises(GenerationError,match='code_commit'):
        planning.verify_code_commit('0'*40)
