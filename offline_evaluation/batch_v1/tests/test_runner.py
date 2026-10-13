"""Durable scheduling/accounting tests. Every transport is a CPU fixture."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from offline_evaluation.resmoke_v2 import adapter
from offline_evaluation.batch_v1 import runner, plan as planning
from offline_evaluation.server_v2.core import Grid, grid_to_payload
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json
from offline_evaluation.server_v2 import providers


STONE = {"operations": [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"}]}
PRIOR = {"calls": 30, "retries": 4, "tokens": 354088,
         "cost_usd_upper_bound": 0.68679325, "unknown_usage_attempts": 0}


def response(provider, text=None, **changes):
    text = json.dumps(STONE) if text is None else text
    value = {"id": "fixture_response", "model": adapter.MODELS[provider],
             "usage": {"input_tokens": 10, "output_tokens": 20}}
    if provider == "openai":
        value.update(model="gpt-5-mini-2025-08-07", status="completed",
                     output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}])
    else:
        value.update(stop_reason="end_turn", content=[{"type": "text", "text": text}])
    value.update(changes)
    return json.dumps(value).encode()


@pytest.fixture
def fixture_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture_only_never_live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture_only_never_live")
    monkeypatch.setattr(adapter, "http_transport", lambda *_: pytest.fail("live transport forbidden"))
    monkeypatch.setattr(providers, "http_transport", lambda *_: pytest.fail("live transport forbidden"))
    parent = tmp_path / "parent"
    parent.mkdir()
    write_json(parent / "campaign.json", {"identity": {"runs": {
        p: {"path": "runs/" + p} for p in adapter.MODELS}}})
    write_json(parent / "ledger.json", {"fixture_parent_usage": PRIOR})
    for provider in adapter.MODELS:
        (parent / "runs" / provider).mkdir(parents=True)
        (parent / "runs" / provider / "run.lock").write_bytes(b"immutable parent lock")
        write_json(parent / "runs" / provider / "state.json", {"fixture_state": provider})
    for name in ("run.lock", "budget.lock"):
        (parent / name).write_bytes(b"immutable parent lock")
    before = {p.relative_to(parent).as_posix(): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    parent_hashes = {name: runner.digest(data) for name, data in before.items()}
    def verify(identity):
        directory = Path(identity["parent"]["campaign_directory"])
        actual = {p.relative_to(directory).as_posix(): runner.digest(p.read_bytes())
                  for p in directory.rglob("*") if p.is_file()}
        if actual != parent_hashes:
            raise GenerationError("parent_changed; replan required")
    monkeypatch.setattr(runner, "_verify_parent", verify)
    from offline_evaluation.server_v2.io_utils import ROOT
    source = read_json(ROOT / 'reports/server_only/goal004/official_pricing.json')
    pricing = {p: source[p] for p in adapter.MODELS}
    for provider, model in adapter.MODELS.items():
        pricing[provider]['verified_on'] = runner.utc_now()[:10]
    document = {'verified_on': runner.utc_now()[:10], 'verified_at_utc': runner.utc_now(),
                'retrieval': 'OFFLINE_FIXTURE_ONLY', 'research_api_requests_sent': 0, **pricing}
    gt = Grid({(0, 0, 0): "stone"})
    scenes = tuple(f"fixture_building_{i:03d}" for i in range(10))
    kinds = [('O', p) for p in ('openai', 'anthropic', 'gemini')] + [('J', p) for p in ('openai', 'anthropic', 'gemini')] + [('P', None), ('E', None)]
    schedule = [{"input_id": scene + condition + str(description), "scene_id": scene, "condition": condition,
                 "description_model": description, "provider": provider, "model": adapter.MODELS[provider],
                 "input_text": scene + " " + condition, 'input_sha256': 'fixture'}
                for scene in scenes for condition, description in kinds for provider in adapter.MODELS]
    for index, item in enumerate(schedule):
        item['schedule_index'] = index
    path = tmp_path / 'plan.json'
    identity = {"protocol_id": "fixture_goal005", "provenance": "fixture_only", 'plan_path': str(path),
                "enabled_directory": str(tmp_path / "enabled"), "authority_path": str(tmp_path / "authority.json"),
                "parent": {"campaign_directory": str(parent), "usage": dict(PRIOR),
                    'lock_paths': [str(p) for p in parent.rglob('*.lock')]},
                "schedule": schedule, "prompt": "Fixture common prompt.", "pricing": pricing,
                'pricing_document_hash': object_hash(document), 'pricing_document': document,
                "budgets": planning.budgets(pricing), 'batch_scene_ids': list(scenes),
                'evaluation_scene_ids': list(scenes) + ['fixture_rest'] * 50,
                'scope': planning.SCOPE, 'schema_sha256': 'fixture', 'prompt_sha256': 'fixture',
                'reservation_policy': 'fixture full context',
                "gt_by_scene": {s: {"grid_sha256": object_hash(grid_to_payload(gt))} for s in scenes}}
    plan = {"identity": identity, "identity_hash": object_hash(identity), "api_allowed": False, "status": "FIXTURE_DISABLED"}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    def load(path):
        value = read_json(Path(path))
        if object_hash(value["identity"]) != value["identity_hash"]:
            raise GenerationError("fixture_plan_hash_mismatch")
        if str(path) != value['identity']['plan_path']:
            raise GenerationError('canonical_plan_path_mismatch')
        return value
    monkeypatch.setattr(runner, "_plan", load)
    monkeypatch.setattr(planning, 'load_plan', load)
    return {"path": path, "plan": plan, "before": before, "parent": parent,
            "grids": {s: gt for s in scenes}}


def approval(plan):
    return {"manual_confirmed": True, "plan_identity_hash": plan["identity_hash"],
            "enabled_directory": plan["identity"]["enabled_directory"],
            "approved_at_utc": runner.utc_now(), "confirmation_channel": "/dev/tty",
            "scope": runner.APPROVAL_SCOPE}


def enable(fixture):
    return runner.initialize_enabled(fixture["path"], approval=approval(fixture["plan"]))


def transports(calls, handler=None):
    def factory(provider):
        def send(request, timeout):
            payload = json.loads(request.data)
            calls.append((provider, payload))
            return handler(provider, payload, len(calls)) if handler else response(provider)
        return send
    return {p: factory(p) for p in adapter.MODELS}



def test_permission_disabled_and_bad_approval_send_zero(fixture_plan):
    calls = []
    with pytest.raises(GenerationError, match='permission'):
        runner.run(fixture_plan['path'], transports=transports(calls))
    with pytest.raises(GenerationError, match='not_enabled'):
        runner.run(fixture_plan['path'], permission=True, transports=transports(calls))
    with pytest.raises(GenerationError, match='fresh_exact_scope'):
        runner.initialize_enabled(fixture_plan['path'], approval={'manual_confirmed': True})
    assert calls == []


def test_fixed160_complete_interleaved_persistent_and_no_repeat(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def inspect(provider, payload, count):
        entry = read_json(directory / 'ledger.json')['entries'][-1]
        assert entry['status'] == 'reserved' and entry['usage'] is None
        assert entry['charged_output_tokens'] == 32768
        return response(provider)
    wire = transports(calls, inspect)
    result = runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert result['status'] == 'COMPLETE_BATCH160' and result['experiment_complete']
    assert result['used']['combined']['calls'] == 190
    assert result['used']['combined']['retries'] == 4
    assert result['used']['combined']['tokens'] == PRIOR['tokens'] + 4800
    assert result['used']['combined']['cost_usd_upper_bound'] == pytest.approx(.69899325)
    assert [p for p, _ in calls] == ['openai', 'anthropic'] * 80
    assert runner.run(fixture_plan['path'], permission=True, transports=wire) == result
    assert len(calls) == 160
    after = {p.relative_to(fixture_plan['parent']).as_posix(): p.read_bytes()
             for p in fixture_plan['parent'].rglob('*') if p.is_file()}
    assert after == fixture_plan['before']


def test_settled_interruption_resume_only_unattempted_and_evaluate_partial(fixture_plan, monkeypatch):
    directory = enable(fixture_plan)
    calls = []
    original = runner.write_json
    def interrupt(path, data):
        original(path, data)
        if path == directory / 'ledger.json' and len(data['entries']) == 4 and data['entries'][-1]['status'] == 'success':
            raise KeyboardInterrupt
    monkeypatch.setattr(runner, 'write_json', interrupt)
    wire = transports(calls)
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan['path'], permission=True, transports=wire)
    monkeypatch.setattr(runner, 'write_json', original)
    monkeypatch.setattr(runner, 'utc_now', lambda: '2099-01-02T00:00:00Z')
    # Historical CPU evaluation is independent of pricing date.
    summary = runner.evaluate_and_summarize(fixture_plan['path'], fixture_plan['grids'])
    assert summary['status'] == 'PARTIAL' and summary['not_run_inputs'] == 156
    rows = read_json(directory / 'evaluation.json')['rows']
    assert len(rows) == 160 and all(row['metrics'] is None for row in rows[4:])
    assert summary['inference_deferred_until_fixed60_complete']
    monkeypatch.undo()
    # A separate test below verifies actual cross-day pricing refresh/resume.


@pytest.mark.parametrize('kind', ['timeout', 'auth_failure', 'billing_failure', 'model_unavailable', 'schema_unsupported'])
def test_unknown_usage_halts_without_resend(fixture_plan, kind):
    directory = enable(fixture_plan)
    calls = []
    def fail(*args):
        raise providers.ProviderError(kind)
    wire = transports(calls, fail)
    result = runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert result['status'] == 'PARTIAL' and result['not_run_inputs'] == 159
    assert result['used']['new']['unknown_usage_attempts'] == 1
    assert result['used']['combined']['cost_usd_upper_bound'] == .85232925
    runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert len(calls) == 1
    assert read_json(directory / 'ledger.json')['entries'][0]['billing'] == 'worst_case_reservation'


def test_crash_uncertain_reservation_never_resends(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def crash(*args):
        raise KeyboardInterrupt
    wire = transports(calls, crash)
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert read_json(directory / 'ledger.json')['entries'][0]['status'] == 'reserved'
    result = runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert result['halt_reason'] == 'unreconciled_prior_reservation' and len(calls) == 1


def test_response_identity_failure_known_charge_still_halts(fixture_plan):
    enable(fixture_plan)
    calls = []
    result = runner.run(fixture_plan['path'], permission=True,
        transports=transports(calls, lambda p, *_: response(p, model='unapproved_model')))
    assert result['halt_reason'] == 'response_identity_failure' and len(calls) == 1


def test_parent_lock_and_parent_change_stop_before_send(fixture_plan):
    enable(fixture_plan)
    calls = []
    path = fixture_plan['parent'] / 'runs/openai/run.lock'
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(GenerationError, match='parent_generation_locked'):
            runner.run(fixture_plan['path'], permission=True, transports=transports(calls))
    (fixture_plan['parent'] / 'ledger.json').write_text('changed')
    with pytest.raises(GenerationError, match='parent_changed'):
        runner.run(fixture_plan['path'], permission=True, transports=transports(calls))
    assert calls == []


def test_new_directory_same_lineage_cannot_reset_budget(fixture_plan):
    enable(fixture_plan)
    new = deepcopy(fixture_plan['plan'])
    path = fixture_plan['path'].with_name('second-plan.json')
    new['identity']['plan_path'] = str(path)
    new['identity']['enabled_directory'] += '-second'
    new['identity_hash'] = object_hash(new['identity'])
    write_json(path, new)
    with pytest.raises(GenerationError, match='lineage_already_claimed'):
        runner.initialize_enabled(path, approval=approval(new))
    assert not Path(new['identity']['enabled_directory']).exists()


def test_budget_insufficient_before_post_and_never_automatically_expands(fixture_plan):
    identity = deepcopy(fixture_plan['plan']['identity'])
    identity['parent']['usage']['cost_usd_upper_bound'] = 9.9
    assert not runner._can_reserve(identity, {'entries': []}, (400000, 32768, .165536))
    identity['parent']['usage']['cost_usd_upper_bound'] = 9.834464
    assert runner._can_reserve(identity, {'entries': []}, (400000, 32768, .165536))
    # This fixture forces a zero-send budget halt without changing real limits.
    plan = fixture_plan['plan']
    plan['identity']['budgets']['max_cost_usd'] = .7
    plan['identity_hash'] = object_hash(plan['identity'])
    write_json(fixture_plan['path'], plan)
    directory = enable(fixture_plan)
    calls = []
    result = runner.run(fixture_plan['path'], permission=True, transports=transports(calls))
    assert result['halt_reason'] == 'budget_insufficient' and calls == []
    assert read_json(directory / 'ledger.json')['entries'] == []


def test_crossday_reverification_preserves_approval_and_charge_then_resumes(fixture_plan, monkeypatch):
    directory = enable(fixture_plan)
    calls = []
    original = runner.write_json
    def stop_after_four(path, value):
        original(path, value)
        if path == directory / 'ledger.json' and len(value['entries']) == 4 and value['entries'][-1]['status'] == 'success':
            raise KeyboardInterrupt
    monkeypatch.setattr(runner, 'write_json', stop_after_four)
    wire = transports(calls)
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan['path'], permission=True, transports=wire)
    monkeypatch.setattr(runner, 'write_json', original)
    before = {name: (directory / name).read_bytes() for name in ['approval.json', 'enabled.json', 'ledger.json']}
    authority_before = Path(fixture_plan['plan']['identity']['authority_path']).read_bytes()
    old_entries = read_json(directory / 'ledger.json')['entries']
    # Control all validator clocks, without modifying original production globals.
    old = fixture_plan['plan']['identity']['pricing']['openai']['verified_on']
    import datetime
    today = (datetime.date.fromisoformat(old) + datetime.timedelta(days=1)).isoformat()
    real_validate = planning.validate_pricing
    def validate(pricing, *, fresh=False):
        real_validate(pricing, fresh=False)
        if fresh and any(p['verified_on'] != today for p in pricing.values()):
            raise GenerationError('pricing_requires_today_verification')
    monkeypatch.setattr(planning, 'validate_pricing', validate)
    monkeypatch.setattr(planning, 'utc_now', lambda: today + 'T12:00:00Z')
    with pytest.raises(GenerationError, match='today_verification'):
        runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert len(calls) == 4
    document = deepcopy(fixture_plan['plan']['identity']['pricing_document'])
    document['verified_on'] = today
    document['verified_at_utc'] = today + 'T11:00:00Z'
    for p in adapter.MODELS:
        document[p]['verified_on'] = today
    price_file = fixture_plan['path'].with_name('reverified.json')
    write_json(price_file, document)
    revision = planning.refresh_pricing(fixture_plan['path'], pricing_file=price_file)
    assert Path(revision['revision_path']).exists()
    assert all((directory / name).read_bytes() == data for name, data in before.items())
    assert Path(fixture_plan['plan']['identity']['authority_path']).read_bytes() == authority_before
    result = runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert result['experiment_complete'] and len(calls) == 160
    assert read_json(directory / 'ledger.json')['entries'][:4] == old_entries
    assert (directory / 'approval.json').read_bytes() == before['approval.json']
    document['openai']['input_usd_per_million'] = 10
    write_json(price_file, document)
    with pytest.raises(GenerationError, match='scope_reapproval'):
        planning.refresh_pricing(fixture_plan['path'], pricing_file=price_file)
    assert len(list((fixture_plan['path'].parent / 'pricing-revisions').iterdir())) == 1


def test_known_usage_model_failure_continues_without_retry_and_empty_separate(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def answer(provider, payload, count):
        text = ('{"operations":[{"op":"set","x":0,"y":0,"z":0,"block":"quartz (white)"}]}'
                if count == 1 else '{"operations":[]}' if count == 2 else None)
        return response(provider, text)
    result = runner.run(fixture_plan['path'], permission=True, transports=transports(calls, answer))
    assert result['experiment_complete'] and len(calls) == 160
    entries = read_json(directory / 'ledger.json')['entries']
    assert entries[0]['failure_kind'] == 'validation_failure'
    assert entries[0]['schema_success'] and entries[0]['failure_stage'] == 'local_validator'
    assert entries[1]['status'] == 'success' and entries[1]['empty_output']
    assert all(not e['retry'] for e in entries)


def test_rolled_back_ledger_cannot_reuse_reserved_schedule(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def fail(*args):
        raise providers.ProviderError('timeout')
    wire = transports(calls, fail)
    runner.run(fixture_plan['path'], permission=True, transports=wire)
    ledger = read_json(directory / 'ledger.json')
    ledger['entries'] = []
    write_json(directory / 'ledger.json', ledger)
    with pytest.raises(GenerationError, match='reservation_claim_ledger_mismatch'):
        runner.run(fixture_plan['path'], permission=True, transports=wire)
    assert len(calls) == 1


@pytest.mark.parametrize('symlink', [False, True])
def test_disabled_evaluation_creates_no_enabled_or_other_files(fixture_plan, symlink):
    directory = Path(fixture_plan['plan']['identity']['enabled_directory'])
    if symlink:
        foreign = fixture_plan['path'].parent / 'foreign-existing-directory'
        foreign.mkdir()
        (foreign / 'sentinel').write_bytes(b'unchanged')
        directory.symlink_to(foreign, target_is_directory=True)
    before = {str(p): p.read_bytes() for p in fixture_plan['path'].parent.rglob('*')
              if p.is_file() and not p.is_symlink()}
    expected = 'symlink_or_traversal' if symlink else 'run_not_enabled'
    with pytest.raises(GenerationError, match=expected):
        runner.evaluate_and_summarize(fixture_plan['path'], fixture_plan['grids'])
    after = {str(p): p.read_bytes() for p in fixture_plan['path'].parent.rglob('*')
             if p.is_file() and not p.is_symlink()}
    assert before == after
    if not symlink:
        assert not directory.exists()
    assert not Path(fixture_plan['plan']['identity']['authority_path']).exists()


def test_disabled_pricing_refresh_does_not_activate_or_reserve(fixture_plan):
    identity = fixture_plan['plan']['identity']
    document = deepcopy(identity['pricing_document'])
    path = fixture_plan['path'].with_name('reviewed-prices.json')
    write_json(path, document)
    before_plan = fixture_plan['path'].read_bytes()
    result = planning.refresh_pricing(fixture_plan['path'], pricing_file=path)
    assert Path(result['revision_path']).is_file()
    assert fixture_plan['path'].read_bytes() == before_plan
    assert not Path(identity['enabled_directory']).exists()
    assert not Path(identity['authority_path']).exists()
    assert list(fixture_plan['path'].parent.rglob('ledger.json')) == [fixture_plan['parent'] / 'ledger.json']
    assert not list(fixture_plan['path'].parent.rglob('reservations'))
