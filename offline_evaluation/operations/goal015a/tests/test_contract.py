"""Independent continuation-contract checks using isolated synthetic parents.

No production approval, provider import, request, model, or GitHub access occurs.
"""
from copy import deepcopy
from decimal import Decimal
import fcntl
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from offline_evaluation.operations.goal015a import runtime as r, evaluate as e
from offline_evaluation.operations.goal015a.tests.test_runtime import fixture_plan, MockAdapter


def _execute(path, adapter, **kwargs):
    options = dict(pricing_evidence=r.read_json(path)['pricing_evidence'], adapter=adapter,
                   confirmation=lambda _: True,
                   checkpoint_callback=lambda *args: {'status': 'LOCAL_VERIFIED', 'fixture_only': True})
    options.update(kwargs)
    return r.execute(path, **options)


def _ledger(path):
    return r.read_json(path.parent / 'enabled/ledger.json')


def _assert_locked(path):
    with Path(path).open('rb') as stream:
        with pytest.raises(BlockingIOError):
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_exact_scope_and_sole_approval_function(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path); plan = r.verify_plan(path)
    assert [row['call_id'] for row in plan['schedule']] == r.CALL_IDS
    assert len(plan['schedule']) == 8 and len({row['paired_block_id'] for row in plan['schedule']}) == 4
    for start in range(0, 8, 2):
        pair = plan['schedule'][start:start + 2]
        assert pair[0]['paired_block_id'] == pair[1]['paired_block_id']
        assert {row['condition'] for row in pair} == {'SUMMARY_ONLY', 'FULL_RECORD'}
    reserve = sum((Decimal(row['reserve_usd']) for row in plan['schedule']), Decimal(0))
    assert reserve == Decimal('2.117504')
    assert Decimal(plan['parent_known_cost_usd']) + reserve == Decimal('6.55516100')
    assert Decimal('7.00') - Decimal(plan['parent_known_cost_usd']) == Decimal('2.56234300')
    assert r.approval_text(plan) == (f"APPROVE GOAL015A {plan['plan_id']} PARENT {plan['parent_plan_id']} "
        'COMBINED USD 7.00 ADDITIONAL CAP USD 2.00 REMAINING 8 CALLS TOTAL 80 RETRY 0')
    monkeypatch.setattr(r, 'approval_text', lambda _: 'one authoritative approval function')
    assert r.preview(path)['approval_text'] == 'one authoritative approval function'
    assert not (path.parent / 'enabled').exists()


def test_both_locks_and_whole_pair_reservation_cover_send_and_checkpoint(tmp_path):
    path = fixture_plan(tmp_path); plan = r.read_json(path); events = []
    parent_lock = plan['parent']['execution_lock']['path']
    parent_lock_bytes = Path(parent_lock).read_bytes()
    class Observed(MockAdapter):
        def send_request(self, *args, **kwargs):
            _assert_locked(parent_lock); _assert_locked(path.parent / 'execution.lock')
            saved = _ledger(path)
            if self.calls == 0:
                assert saved['calls'][0]['state'] == 'SENT_PENDING'
                assert saved['calls'][1]['state'] == 'NOT_SENT'
                assert [row['reservation_usd'] for row in saved['calls'][:2]] == ['0.363840'] * 2
                assert all(row['reservation_usd'] == '0' for row in saved['calls'][2:])
                assert Decimal(r.accounting(plan, saved)['combined_committed_usd']) == Decimal('5.16533700')
                assert Decimal(r.accounting(plan, saved)['parent_known_cost_usd']) == Decimal('4.43765700')
            events.append('send')
            return super().send_request(*args, **kwargs)
    def checkpoint(*args):
        _assert_locked(parent_lock); _assert_locked(path.parent / 'execution.lock')
        events.append('checkpoint')
        return {'status': 'LOCAL_VERIFIED', 'fixture_only': True}
    adapter = Observed(); result = _execute(path, adapter, checkpoint_callback=checkpoint)
    assert result['status'] == 'COMPLETE' and events == ['send'] * 8 + ['checkpoint']
    assert Path(parent_lock).read_bytes() == parent_lock_bytes


@pytest.mark.parametrize('which', ['parent', 'child'])
def test_existing_lock_prevents_approval_and_dispatch(tmp_path, which):
    path = fixture_plan(tmp_path); plan = r.read_json(path); adapter = MockAdapter(); prompts = []
    lock_path = Path(plan['parent']['execution_lock']['path']) if which == 'parent' else path.parent / 'execution.lock'
    if which == 'child': lock_path.write_bytes(b'fixture existing child lock')
    with lock_path.open('rb') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(r.RuntimeErrorSafe, match='execution already active'):
            _execute(path, adapter, confirmation=lambda _: prompts.append(True) or True)
    assert adapter.calls == 0 and prompts == [] and not (path.parent / 'enabled').exists()


def test_expiry_between_siblings_stops_before_second_request_retains_hold(tmp_path, monkeypatch):
    path = fixture_plan(tmp_path); adapter = MockAdapter(); original = r.pricing_fresh
    def expires(plan, evidence):
        if adapter.calls:
            raise r.RuntimeErrorSafe('fixture pricing evidence expired')
        return original(plan, evidence)
    monkeypatch.setattr(r, 'pricing_fresh', expires)
    result = _execute(path, adapter)
    assert adapter.calls == 1 and result['status'] == 'SAFE_STOP'
    saved = _ledger(path)
    assert saved['calls'][0]['state'] == 'SETTLED' and saved['calls'][1]['state'] == 'NOT_SENT'
    assert saved['calls'][1]['reservation_usd'] == '0.363840'
    assert Decimal(result['budget']['unsettled_and_unknown_holds_usd']) == Decimal('0.363840')
    with pytest.raises(r.RuntimeErrorSafe, match='expired'):
        _execute(path, adapter, resume=True)
    assert adapter.calls == 1


def test_raw_is_saved_and_bound_before_usage_settlement(tmp_path):
    path = fixture_plan(tmp_path); inspected = []
    class Observed(MockAdapter):
        def classify_usage(self, *args):
            row = _ledger(path)['calls'][self.calls - 1]
            assert row['state'] == 'SENT_PENDING' and row['known_cost_usd'] is None
            assert row['raw'] == r.descriptor(row['raw']['path'])
            assert Path(row['raw']['path']).read_bytes()
            inspected.append(row['call_id'])
            return super().classify_usage(*args)
    adapter = Observed(); _execute(path, adapter)
    assert inspected == r.CALL_IDS


@pytest.mark.parametrize('field,value', [
    ('response_received', True), ('completed_response', True),
    ('started_at_utc', 'fixture'), ('finished_at_utc', 'fixture'),
    ('response_text_path', {'path': '/not/a/real/response', 'bytes': 1, 'sha256': '0' * 64}),
])
def test_not_sent_transmission_evidence_cannot_be_resumed(tmp_path, field, value):
    path = fixture_plan(tmp_path); adapter = MockAdapter(unknown_at=1); _execute(path, adapter)
    saved = _ledger(path); saved['calls'][1][field] = value
    r._save(path.parent / 'enabled/ledger.json', saved)
    with pytest.raises(r.RuntimeErrorSafe, match='NOT_SENT has transmission evidence'):
        _execute(path, adapter, resume=True)
    assert adapter.calls == 1


@pytest.mark.parametrize('suffix', ['.raw.json', '.text.txt'])
def test_orphan_response_file_prevents_restored_unsent_resend(tmp_path, suffix):
    path = fixture_plan(tmp_path); adapter = MockAdapter(unknown_at=1); _execute(path, adapter)
    orphan = path.parent / 'enabled/responses' / (r.CALL_IDS[1] + suffix)
    orphan.write_bytes(b'previous durable response; ledger restoration is unsafe')
    with pytest.raises(r.RuntimeErrorSafe, match='existing response files'):
        _execute(path, adapter, resume=True)
    assert adapter.calls == 1 and orphan.read_bytes().startswith(b'previous durable')


@pytest.mark.parametrize('field,value', [
    ('plan_id', '0' * 64), ('parent_plan_id', '0' * 64),
    ('canonical_root', '/copied/root'), ('scope_calls', 80), ('max_retries', 1), ('status', 'NOT_APPROVED'),
])
def test_authority_fields_cannot_be_replaced_on_resume(tmp_path, field, value):
    path = fixture_plan(tmp_path); adapter = MockAdapter(unknown_at=1); _execute(path, adapter)
    authority_path = path.parent / 'enabled/authority.json'; authority = r.read_json(authority_path)
    authority[field] = value; r._save(authority_path, authority)
    with pytest.raises(r.RuntimeErrorSafe, match='authority'):
        _execute(path, adapter, resume=True)
    assert adapter.calls == 1


def test_contract_failure_remains_resume_blocker_without_global_flag(tmp_path):
    path = fixture_plan(tmp_path)
    class Mismatch(MockAdapter):
        def parse_response(self, *args):
            response = super().parse_response(*args)
            response.failure_kind = 'response_model_mismatch'
            response.model = 'wrong snapshot'
            return response
    adapter = Mismatch(unknown_at=1); _execute(path, adapter)
    saved = _ledger(path); assert saved['requires_new_explicit_approval']
    saved.pop('requires_new_explicit_approval'); r._save(path.parent / 'enabled/ledger.json', saved)
    with pytest.raises(r.RuntimeErrorSafe, match='contract failure'):
        _execute(path, adapter, resume=True)
    assert adapter.calls == 1


@pytest.mark.parametrize('mutation', ['parent_ledger', 'payload', 'copied_plan'])
def test_changed_frozen_inputs_reject_before_approval(tmp_path, mutation):
    path = fixture_plan(tmp_path); plan = r.read_json(path); adapter = MockAdapter(); prompts = []
    if mutation == 'copied_plan':
        other = tmp_path / 'copied-plan.json'; other.write_bytes(path.read_bytes()); path = other
    else:
        target = Path(plan['parent']['ledger']['path'] if mutation == 'parent_ledger' else plan['schedule'][0]['payload']['path'])
        target.write_bytes(target.read_bytes() + b' ')
    with pytest.raises((r.RuntimeErrorSafe, ValueError)):
        _execute(path, adapter, confirmation=lambda _: prompts.append(True) or True)
    assert adapter.calls == 0 and prompts == []
    assert not (Path(plan['canonical_root']) / 'enabled').exists()


def test_fixture_cannot_fall_through_to_real_transport(tmp_path):
    path = fixture_plan(tmp_path); plan = r.read_json(path)
    with pytest.raises(r.RuntimeErrorSafe, match='mock adapter'):
        r.execute(path, pricing_evidence=plan['pricing_evidence'], confirmation=lambda _: True)
    assert not (path.parent / 'enabled').exists()


@pytest.mark.parametrize('cap', ['7', '7.0', '7.000', '9.00'])
def test_cap_is_fixed_exact_combined_seven_not_seven_additional(tmp_path, cap):
    path = fixture_plan(tmp_path); plan = r.read_json(path); adapter = MockAdapter()
    plan['proposed_budget_usd'] = cap; plan['plan_id'] = r.identity(plan); r._save(path, plan)
    with pytest.raises(r.RuntimeErrorSafe, match='combined cap'):
        _execute(path, adapter)
    assert adapter.calls == 0 and not (path.parent / 'enabled').exists()


@pytest.mark.parametrize('mutation', ['duplicate', 'reordered', 'payload_identity', 'extra'])
def test_merged_selection_rejects_changed_residual_identity(tmp_path, mutation):
    path = fixture_plan(tmp_path); plan = r.read_json(path); _execute(path, MockAdapter())
    parent = r.verify_parent(plan); saved = _ledger(path)
    if mutation == 'duplicate': saved['calls'][1] = deepcopy(saved['calls'][0])
    elif mutation == 'reordered': saved['calls'][0], saved['calls'][1] = saved['calls'][1], saved['calls'][0]
    elif mutation == 'payload_identity': saved['calls'][0]['payload']['sha256'] = '0' * 64
    else: saved['calls'].append(deepcopy(saved['calls'][0]))
    with pytest.raises((ValueError, r.RuntimeErrorSafe)):
        e.select_calls(parent['plan'], parent['ledger'], plan, saved)


def test_merge_is_one_to_one_detached_and_does_not_impute_unknown(tmp_path):
    path = fixture_plan(tmp_path); plan = r.read_json(path); adapter = MockAdapter(unknown_at=1)
    _execute(path, adapter); _execute(path, adapter, resume=True)
    parent = r.verify_parent(plan); saved = _ledger(path)
    selected, sources = e.select_calls(parent['plan'], parent['ledger'], plan, saved)
    assert len(selected) == len({row['call_id'] for row in selected}) == 80
    assert selected[:72] == parent['ledger']['calls'][:72]
    assert selected[72]['state'] == 'UNKNOWN_USAGE' and selected[72]['known_cost_usd'] is None
    assert [row['selected_source'] for row in sources] == ['PARENT'] * 72 + ['CHILD'] * 8
    selected[0]['state'] = 'edited copy'; selected[72]['state'] = 'edited copy'
    assert parent['ledger']['calls'][0]['state'] == 'SETTLED' and saved['calls'][0]['state'] == 'UNKNOWN_USAGE'
    assert adapter.calls == 8


def test_changed_original_gt_fails_child_cpu_evaluation(tmp_path, monkeypatch):
    from offline_evaluation.operations.goal015a.tests.test_evaluate_checkpoint import fixture_plan as rich_fixture
    path = rich_fixture(tmp_path, monkeypatch); plan = r.read_json(path); adapter = MockAdapter()
    _execute(path, adapter); parent = r.bound(plan['parent']['plan'])
    scene = plan['schedule'][0]['scene_id']
    gt = Path(parent['gt_sources'][scene]['building_directory']) / 'gt/voxels.npy'
    gt.write_bytes(b'changed GT fixture')
    with pytest.raises((ValueError, r.RuntimeErrorSafe), match='GT bytes'):
        e.evaluate_plan(path)
    assert adapter.calls == 8


def test_merged_empty_success_and_unknown_remain_distinct(tmp_path, monkeypatch):
    from offline_evaluation.operations.goal015a.tests.test_evaluate_checkpoint import fixture_plan as rich_fixture
    path = rich_fixture(tmp_path, monkeypatch); plan = r.read_json(path)
    adapter = MockAdapter(unknown_at=1); _execute(path, adapter); _execute(path, adapter, resume=True)
    parent_before = Path(plan['parent']['ledger']['path']).read_bytes()
    summary = e.evaluate_plan(path); rows = r.bound(summary['evaluation_artifact'])['rows']
    assert len(rows) == 80 and len({row['call_id'] for row in rows}) == 80
    assert rows[72]['status'] == 'OPERATIONAL_UNKNOWN' and rows[72]['metrics'] is None
    assert not e.frozen.failure0_eligible(rows[72])
    for row in rows[73:]:
        assert row['status'] == 'PROTOCOL_VALID' and row['protocol_valid'] and row['empty_output']
        assert row['metrics'] is not None and row['failure_kind'] is None
        assert not e.frozen.failure0_eligible(row)
    assert summary['planned_independent_buildings'] == 10 and summary['actual_metric_rows'] == 78
    assert adapter.calls == 8 and Path(plan['parent']['ledger']['path']).read_bytes() == parent_before


def test_save_only_retry_after_unknown_never_dispatches_or_releases_hold(tmp_path, monkeypatch):
    from offline_evaluation.operations.goal015a import checkpoint as c
    from offline_evaluation.operations.goal015a.tests.test_evaluate_checkpoint import fixture_plan as rich_fixture, FakeGithub
    path = rich_fixture(tmp_path, monkeypatch); adapter = MockAdapter(unknown_at=1)
    stopped = _execute(path, adapter); ledger_before = (path.parent / 'enabled/ledger.json').read_bytes()
    assert stopped['status'] == 'SAFE_STOP' and stopped['calls_uncertain'] == 1
    saved = c.create_save_only(path); package = r.bound(saved['checkpoint'])
    github = FakeGithub(package['target_commitish']); github.fail_download = True
    with pytest.raises(RuntimeError, match='fixture interruption'):
        c.upload_checkpoint(saved['path'], tmp_path / 'interrupted-verification', allow_create=True, github=github)
    github.fail_download = False
    verification = c.upload_checkpoint(saved['path'], tmp_path / 'retry-verification', github=github)
    assert verification['status'] == 'GITHUB_VERIFIED' and not verification['uploaded']
    assert adapter.calls == 1 and (path.parent / 'enabled/ledger.json').read_bytes() == ledger_before
    assert Decimal(r.status(path)['budget']['unsettled_and_unknown_holds_usd']) == Decimal('0.727680')
    assert sum(stage == 'release_create' for stage, _, _ in github.calls) == 1
    assert sum(stage == 'asset_upload' for stage, _, _ in github.calls) == 1


@pytest.mark.parametrize('mutation', ['authority_plan', 'authority_approval', 'approval_budget'])
def test_archive_rejects_rehashed_inconsistent_child_authority(tmp_path, monkeypatch, mutation):
    from offline_evaluation.operations.goal015a import checkpoint as c
    from offline_evaluation.operations.goal015a.tests.test_evaluate_checkpoint import fixture_plan as rich_fixture
    path = rich_fixture(tmp_path, monkeypatch); _execute(path, MockAdapter(unknown_at=1))
    saved = c.create_save_only(path); archive = Path(saved['path']) / 'goal015a-state.tar.gz'
    with tarfile.open(archive, 'r:gz') as source:
        contents = {member.name: source.extractfile(member).read() for member in source.getmembers()}
    member = 'child/enabled/approval.json' if mutation == 'approval_budget' else 'child/enabled/authority.json'
    document = json.loads(contents[member])
    if mutation == 'approval_budget': document['approved_combined_budget_usd'] = '9.00'
    elif mutation == 'authority_plan': document['plan']['sha256'] = '0' * 64
    else: document['approval']['sha256'] = '0' * 64
    contents[member] = (json.dumps(document, sort_keys=True) + '\n').encode()
    manifest = json.loads(contents['FILES.json'])
    manifest['files'][member] = {'bytes': len(contents[member]), 'sha256': hashlib.sha256(contents[member]).hexdigest()}
    contents['FILES.json'] = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    tampered = tmp_path / 'internally-inconsistent-state.tar.gz'
    with tarfile.open(tampered, 'w:gz') as target:
        for name, raw in contents.items():
            info = tarfile.TarInfo(name); info.size = len(raw)
            target.addfile(info, io.BytesIO(raw))
    with pytest.raises((ValueError, r.RuntimeErrorSafe)):
        c.verify_archive(tampered, r.descriptor(tampered)['sha256'])
