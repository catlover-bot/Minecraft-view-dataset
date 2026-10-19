"""Isolated synthetic parent72 and child8; no production approval or transport."""
import copy
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from offline_evaluation.operations.goal015a import runtime as r, planning
from offline_evaluation.experiments.goal015.tests.test_runtime import fixture_plan as old_fixture_plan


def fixture_plan(tmp_path):
    parent_path = old_fixture_plan(tmp_path)
    parent = r.read_json(parent_path)
    final = parent['schedule'][72:]
    parent['schedule'][72:] = [final[i] for i in (5, 4, 1, 0, 3, 2, 7, 6)]
    for i, row in enumerate(parent['schedule'], 1):
        row['call_id'] = f'goal015_call_{i:03d}'
        row['reserve_usd'] = {'openai': '0.165536', 'anthropic': '0.363840'}[row['builder']]
    parent['plan_id'] = r.identity(parent)
    parent_path.write_text(json.dumps(parent))
    root = parent_path.parent
    (root / 'execution.lock').write_bytes(b'unchanged fixture parent lock')
    enabled = root / 'enabled'; enabled.mkdir()
    raw = root / 'raw.json'; raw.write_text('{}')
    text = root / 'response.txt'; text.write_text('{"operations":[]}')
    approval = {'status': 'APPROVED', 'plan_id': parent['plan_id'], 'canonical_root': str(root),
                'approved_budget_usd': '5.00', 'scope_calls': 80}
    r._save(enabled / 'approval.json', approval, exclusive=True)
    authority = {'status': 'APPROVED', 'scope_calls': 80, 'max_retries': 0,
                 'plan_id': parent['plan_id'], 'canonical_root': str(root), 'plan': r.descriptor(parent_path),
                 'approval': r.descriptor(enabled / 'approval.json')}
    r._save(enabled / 'authority.json', authority, exclusive=True)
    calls = []
    for i, row in enumerate(parent['schedule']):
        sent = i < 72
        calls.append(dict(row, state='SETTLED' if sent else 'NOT_SENT', reservation_usd=row['reserve_usd'] if sent else '0',
            known_cost_usd='0.061634125' if sent else None, usage_known=sent,
            send_status='POST_RETURNED' if sent else 'NOT_SENT', response_received=sent, completed_response=sent,
            model_output_failure=False, failure_kind=None, raw=r.descriptor(raw) if sent else None,
            response_text_path=r.descriptor(text) if sent else None))
    ledger = {'plan_id': parent['plan_id'], 'status': 'SAFE_STOP', 'calls': calls,
              'halt': {'reason': 'next complete paired block exceeds approved cap'}}
    r._save(enabled / 'ledger.json', ledger, exclusive=True)
    evaluation = root / 'evaluation_80.json'
    r._save(evaluation, {'plan_binding': {'plan_id': parent['plan_id'], 'plan': r.descriptor(parent_path),
            'ledger': r.descriptor(enabled / 'ledger.json')}, 'rows': [{'call_id': row['call_id']} for row in calls]}, exclusive=True)
    archive = root / 'fixture.tar.gz'; archive.write_bytes(b'fixture archive, never uploaded')
    checkpoint = root / 'checkpoint.json'
    r._save(checkpoint, {'plan_id': parent['plan_id'], 'archive_sha256': r.descriptor(archive)['sha256']}, exclusive=True)
    proof = root / 'github.json'
    r._save(proof, {'status': 'GITHUB_VERIFIED', 'draft': True, 'release_id': 1, 'asset_id': 2,
        'archive_sha256': r.descriptor(archive)['sha256'], 'fixture_only': True, 'github_transport': 'FAKE_FIXTURE'}, exclusive=True)
    child = tmp_path / 'child'; child.mkdir()
    planning.prepare_plan(parent_path, child, parent_evaluation=r.descriptor(evaluation), github_verification=r.descriptor(proof),
        parent_checkpoint=r.descriptor(checkpoint), parent_archive=r.descriptor(archive),
        specification=parent['specification'], implementation_hashes=parent['implementation_hashes'],
        pricing_evidence=parent['pricing_evidence'], fixture_only=True)
    return child / 'plan.json'


class MockAdapter:
    def __init__(self, unknown_at=None, fail_at=None, interrupt_at=None):
        self.calls = 0; self.unknown_at = unknown_at; self.fail_at = fail_at; self.interrupt_at = interrupt_at
    def validate_payload(self, builder, payload):
        assert payload == {'mock': True}
    def send_request(self, builder, payload, **kwargs):
        assert kwargs == {'api_allowed': True, 'permission': True, 'timeout': 600}
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt()
        if self.calls == self.fail_at:
            raise OSError('must not be logged secret')
        return json.dumps({'usage_known': self.calls != self.unknown_at}).encode()
    def parse_response(self, builder, raw):
        return SimpleNamespace(text='{"operations":[]}', complete=True, model='fixture', response_id='fixture',
                               failure_kind=None, usage=json.loads(raw))
    def classify_usage(self, builder, usage, pricing):
        return {'usage_known': usage['usage_known'], 'cost_usd': '0.01' if usage['usage_known'] else None}


def execute(path, adapter=None, **kwargs):
    return r.execute(path, pricing_evidence=r.read_json(path)['pricing_evidence'], adapter=adapter or MockAdapter(),
                     confirmation=lambda plan: True,
                     checkpoint_callback=lambda *args: {'status': 'LOCAL_VERIFIED', 'fixture_only': True}, **kwargs)


def test_scope_cost_preview_preserves_parent(tmp_path):
    p = fixture_plan(tmp_path); plan = r.verify_plan(p)
    before = {ref['path']: Path(ref['path']).read_bytes() for ref in plan['parent'].values()}
    out = r.preview(p)
    assert out['planned_calls'] == 8 and out['proposed_combined_cap_usd'] == '7.00' and not out['budget_approved']
    assert plan['costs']['parent_plus_full_remaining_reservation_usd'] == '6.555161000'
    assert not (p.parent / 'enabled').exists() and not (p.parent / 'execution.lock').exists()
    assert all(Path(name).read_bytes() == raw for name, raw in before.items())


def test_child8_complete_and_parent_unchanged(tmp_path):
    p = fixture_plan(tmp_path); plan = r.read_json(p); adapter = MockAdapter()
    before = {ref['path']: Path(ref['path']).read_bytes() for ref in plan['parent'].values()}
    result = execute(p, adapter)
    assert result['status'] == 'COMPLETE' and result['calls_settled'] == adapter.calls == 8
    assert result['budget']['combined_known_cost_usd'] == '4.51765700'
    assert all(Path(name).read_bytes() == raw for name, raw in before.items())
    execute(p, adapter, resume=True)
    assert adapter.calls == 8
    with pytest.raises(r.RuntimeErrorSafe, match='existing child'):
        execute(p, adapter)


@pytest.mark.parametrize('kind', ['unknown', 'transport', 'interrupt'])
def test_uncertain_retained_never_resends(tmp_path, kind):
    p = fixture_plan(tmp_path)
    adapter = MockAdapter(unknown_at=1 if kind == 'unknown' else None,
                          fail_at=1 if kind == 'transport' else None,
                          interrupt_at=1 if kind == 'interrupt' else None)
    first = execute(p, adapter)
    assert first['status'] == 'SAFE_STOP' and first['calls_uncertain'] == 1 and adapter.calls == 1
    assert first['budget']['unsettled_and_unknown_holds_usd'] == '0.727680'
    second = execute(p, adapter, resume=True)
    assert second['status'] == 'PARTIAL_WITH_UNCERTAIN_CALLS' and second['calls_settled'] == 7 and adapter.calls == 8
    assert second['budget']['unsettled_and_unknown_holds_usd'] == '0.363840'
    assert 'must not be logged' not in (p.parent / 'enabled/ledger.json').read_text()


def test_no_approval_no_child_authority(tmp_path):
    p = fixture_plan(tmp_path); adapter = MockAdapter()
    with pytest.raises(r.RuntimeErrorSafe, match='approval refused'):
        r.execute(p, pricing_evidence=r.read_json(p)['pricing_evidence'], adapter=adapter, confirmation=lambda plan: False)
    assert adapter.calls == 0 and not (p.parent / 'enabled').exists() and not (p.parent / 'execution.lock').exists()


def test_parent_changed_during_approval_refused(tmp_path):
    p = fixture_plan(tmp_path); plan = r.read_json(p); adapter = MockAdapter()
    def approve(_):
        path = Path(plan['parent']['ledger']['path']); path.write_bytes(path.read_bytes() + b' ')
        return True
    with pytest.raises(r.RuntimeErrorSafe, match='bound parent'):
        r.execute(p, pricing_evidence=plan['pricing_evidence'], adapter=adapter, confirmation=approve)
    assert not (p.parent / 'enabled').exists() and adapter.calls == 0


@pytest.mark.parametrize('answer', ['exact', 'enter', 'reject', 'eof'])
def test_real_pty_exact_only(tmp_path, answer):
    p = fixture_plan(tmp_path); plan = r.read_json(p); pid, master = pty.fork()
    if pid == 0:
        try:
            r.confirm_tty(plan); os._exit(0)
        except BaseException:
            os._exit(7)
    output = b''; sent = False; deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], .05)
        if ready:
            try: output += os.read(master, 8192)
            except OSError: pass
        if b'> ' in output and not sent:
            data = (r.approval_text(plan) + '\n').encode() if answer == 'exact' else {'enter': b'\n', 'reject': b'no\n', 'eof': b'\x04'}[answer]
            os.write(master, data); sent = True
        done, code = os.waitpid(pid, os.WNOHANG)
        if done:
            os.close(master)
            assert (os.waitstatus_to_exitcode(code) == 0) == (answer == 'exact')
            assert b'combined cap USD 7.00' in output
            assert not (p.parent / 'enabled').exists()
            return
    os.kill(pid, 9); os.waitpid(pid, 0); os.close(master)
    raise AssertionError('PTY timed out')


def test_non_tty_fails_closed(tmp_path):
    p = fixture_plan(tmp_path)
    code = 'from offline_evaluation.operations.goal015a import runtime as r;r.confirm_tty(r.read_json(' + repr(str(p)) + '))'
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, start_new_session=True)
    assert result.returncode != 0 and b'TTY' in result.stderr
    assert not (p.parent / 'enabled').exists()


def test_real_pty_execute8_then_duplicate_refused(tmp_path):
    p = fixture_plan(tmp_path); plan = r.read_json(p); pid, master = pty.fork()
    if pid == 0:
        try:
            adapter = MockAdapter()
            kwargs = {'pricing_evidence': plan['pricing_evidence'], 'adapter': adapter,
                      'checkpoint_callback': lambda *args: {'status': 'LOCAL_VERIFIED', 'fixture_only': True}}
            assert r.execute(p, **kwargs)['calls_settled'] == 8
            try:
                r.execute(p, **kwargs)
            except r.RuntimeErrorSafe as exc:
                assert 'existing child' in str(exc)
            else:
                raise AssertionError('duplicate activation accepted')
            assert adapter.calls == 8
            os._exit(0)
        except BaseException:
            os._exit(7)
    output = b''; sent = False; deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], .05)
        if ready:
            try: output += os.read(master, 8192)
            except OSError: pass
        if b'> ' in output and not sent:
            os.write(master, (r.approval_text(plan) + '\n').encode()); sent = True
        done, code = os.waitpid(pid, os.WNOHANG)
        if done:
            os.close(master)
            assert os.waitstatus_to_exitcode(code) == 0 and output.count(b'Type exactly:') == 1
            assert r.status(p)['calls_settled'] == 8
            return
    os.kill(pid, 9); os.waitpid(pid, 0); os.close(master)
    raise AssertionError('PTY execution timed out')


def test_invalid_known_usage_keeps_reservation(tmp_path):
    p = fixture_plan(tmp_path)
    class Invalid(MockAdapter):
        def classify_usage(self, *args):
            return {'usage_known': True, 'cost_usd': None}
    adapter = Invalid(); out = execute(p, adapter)
    assert out['status'] == 'SAFE_STOP' and out['calls_uncertain'] == 1 and adapter.calls == 1
    row = r.read_json(p.parent / 'enabled/ledger.json')['calls'][0]
    assert row['usage_known'] is False and row['known_cost_usd'] is None
    assert out['budget']['unsettled_and_unknown_holds_usd'] == '0.727680'


@pytest.mark.parametrize('status', [None, 'REVOKED'])
def test_authority_status_required_for_status_and_resume(tmp_path, status):
    p = fixture_plan(tmp_path); adapter = MockAdapter(unknown_at=1)
    execute(p, adapter)
    authority_path = p.parent / 'enabled/authority.json'; authority = r.read_json(authority_path)
    if status is None:
        authority.pop('status')
    else:
        authority['status'] = status
    authority_path.write_text(json.dumps(authority))
    with pytest.raises(r.RuntimeErrorSafe, match='authority binding'):
        r.status(p)
    with pytest.raises(r.RuntimeErrorSafe, match='authority binding'):
        execute(p, adapter, resume=True)
    assert adapter.calls == 1
