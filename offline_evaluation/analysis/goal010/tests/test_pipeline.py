"""Synthetic completion, phase ordering, source and output tamper regressions."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from offline_evaluation.analysis.goal010 import pipeline as p

CODE = 'a' * 40
SPEC = 'b' * 40
HASH = 'c' * 64


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def case(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'; runs.mkdir()
    root = runs / 'goal010-fixture-20261017T010203Z'
    monkeypatch.setattr(p, 'RUNS', runs)
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20*1024**3))
    binding = {'code_commit': CODE, 'specification_commit': SPEC,
               'specification_sha256': HASH, 'code_and_specification_hashes': {'fixture.py': HASH}}
    monkeypatch.setattr(p, 'code_binding', lambda code, spec: deepcopy(binding))
    state = {'loaded': 0, 'active': False, 'entered': 0, 'exited': 0, 'executed': [], 'value': 'unchanged'}
    @contextmanager
    def locks():
        state['entered'] += 1; state['active'] = True
        with p.guard.audit8.old_audit.no_python_network():
            yield [1, 2]
        state['active'] = False; state['exited'] += 1
    monkeypatch.setattr(p.guard, 'locked_sources', locks)
    def load():
        assert state['active']; state['loaded'] += 1
        return {'synthetic': state['value']}
    monkeypatch.setattr(p.guard, 'load_verified_sources', load)
    def execute(number, package, outroot, gate, bound):
        assert state['active']; state['executed'].append(number)
        # Exercise the installed Python blockade without creating an OS socket.
        with pytest.raises(p.guard.audit8.old_audit.AuditError):
            socket.socket.connect(None, ('127.0.0.1', 1))
        out = outroot / p.DIRECTORIES[number]; out.mkdir()
        p.write_json(out / 'synthetic.json', {'number': number})
        if number == 0:
            p.write_json(out / 'source_gate.json', {'status': 'PASS', 'input_kind': 'verified_sources',
                'phase0_binding': 'PASS', 'definitions_frozen': True, 'specification_sha256': HASH,
                'code_commit': CODE, 'goal007_manifest_sha256': p.guard.audit8.MANIFEST_SHA256})
            p.write_json(out / 'source_inventory_receipt.json', {'source_package_sha256': p.digest(package)})
        else:
            assert gate['status'] == 'PASS'
        return {'status': 'PASS'}
    monkeypatch.setattr(p, 'execute_phase', execute)
    return root, state, binding, execute


def run(root, phase):
    return p.run_phase(phase, root, CODE, SPEC)


def test_all_six_phases_complete_with_real_path_json_boundary(case):
    root, state, binding, _ = case
    for phase in range(6):
        receipt = run(root, phase)
        saved = read(root / f'PHASE{phase}_RECEIPT.json')
        assert receipt == saved and saved['status'] == 'PASS'
        assert saved['source_before_after_unchanged'] is True
        assert saved['research_api_requests_sent'] == saved['api_key_queries'] == 0
        assert len(saved['prior_receipts']) == phase
        assert saved['source_lock_count'] == 2
        assert saved['python_network_connections_disabled'] is True
        assert saved['OS_level_isolation_guaranteed'] is False
        for name, descriptor in saved['products'].items():
            path = root / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['sha256']
    assert state['executed'] == list(range(6))
    assert state['loaded'] == 12 and state['entered'] == state['exited'] == 6


def test_skip_phase_fails_before_source_or_analysis(case):
    root, state, _, _ = case
    run(root, 0)
    before = state['loaded']
    with pytest.raises(Exception): run(root, 2)
    assert state['loaded'] == before and state['executed'] == [0]


@pytest.mark.parametrize('phase', [0, 1])
def test_no_overwrite_completed_phase(case, phase):
    root, state, _, _ = case
    for number in range(phase+1): run(root, number)
    before = {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises((ValueError, FileExistsError)): run(root, phase)
    assert before == {str(path): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_previous_product_tamper_prevents_source_load(case):
    root, state, _, _ = case; run(root, 0)
    (root / p.DIRECTORIES[0] / 'synthetic.json').write_text('{}')
    before = state['loaded']
    with pytest.raises(ValueError, match='hash mismatch'): run(root, 1)
    assert state['loaded'] == before


def test_different_commit_prevents_next_phase(case, monkeypatch):
    root, state, binding, _ = case; run(root, 0)
    changed = deepcopy(binding); changed['code_commit'] = 'd'*40
    monkeypatch.setattr(p, 'code_binding', lambda *args: changed)
    with pytest.raises(ValueError, match='different code'): run(root, 1)
    assert state['executed'] == [0]


def test_source_changed_after_phase_preserves_stop_and_partial_output(case, monkeypatch):
    root, state, _, execute = case
    def changed(*args):
        result = execute(*args); state['value'] = 'changed'; return result
    monkeypatch.setattr(p, 'execute_phase', changed)
    with pytest.raises(ValueError, match='sources changed'): run(root, 0)
    assert read(root/'PHASE0_STOP.json')['status'] == 'STOP'
    assert not (root/'PHASE0_RECEIPT.json').exists()
    assert (root/p.DIRECTORIES[0]/'synthetic.json').is_file()
    with pytest.raises(FileExistsError): run(root, 0)


def test_changed_source_between_phases_stops_before_analysis(case):
    root, state, _, _ = case; run(root, 0); state['value'] = 'changed'
    with pytest.raises(ValueError, match='differs from Phase0'): run(root, 1)
    assert state['executed'] == [0]
    assert read(root/'PHASE1_STOP.json')['status'] == 'STOP'
    assert not (root/'PHASE1_RECEIPT.json').exists()


def test_failure_reason_cannot_leak_exception_payload(case, monkeypatch, capsys):
    root, state, _, _ = case
    def fail(*args): raise RuntimeError('DO_NOT_LEAK_RAW_RESPONSE_OR_SECRET')
    monkeypatch.setattr(p, 'execute_phase', fail)
    status = p.main(['--phase','0','--run-root',str(root),'--code-commit',CODE,'--specification-commit',SPEC])
    assert status == 5
    assert 'DO_NOT_LEAK' not in capsys.readouterr().out
    assert 'DO_NOT_LEAK' not in (root/'PHASE0_STOP.json').read_text()


def test_exit_process_guard_failure_cannot_record_pass(case, monkeypatch):
    root, state, _, _ = case
    @contextmanager
    def failing_exit():
        state['active'] = True
        with p.guard.audit8.old_audit.no_python_network(): yield [1]
        raise ValueError('post-lock process audit failed')
    monkeypatch.setattr(p.guard, 'locked_sources', failing_exit)
    with pytest.raises(ValueError): run(root, 0)
    assert not (root/'PHASE0_RECEIPT.json').exists()
    assert read(root/'PHASE0_STOP.json')['status'] == 'STOP'


@pytest.mark.parametrize('name', ['other-20261017T010203Z','goal010-without-date','goal010-20261017T010203Z/child'])
def test_unsafe_root_names_rejected(case, name):
    root, *_ = case
    with pytest.raises(ValueError): p.safe_root(root.parent/name, 0)


def test_symlink_and_low_disk_rejected(case, tmp_path, monkeypatch):
    root, *_ = case
    root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'): p.safe_root(root, 0)
    root.unlink()
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match='GiB'): p.safe_root(root, 0)


def test_prediction_arrays_are_verified_through_frozen_manifest(case):
    root, *_ = case
    directory=root/'phase1_baselines'; (directory/'predictions').mkdir(parents=True)
    array=directory/'predictions/fixture.npy'; array.write_bytes(b'fixture array bytes')
    descriptor={'path':array.name,'sha256':hashlib.sha256(array.read_bytes()).hexdigest(),'bytes':array.stat().st_size}
    p.write_json(directory/'baseline_predictions_manifest.json',{'predictions':[{'artifacts':{'coordinates':descriptor}}]})
    p.verify_predictions(root)
    assert 'phase1_baselines/predictions/fixture.npy' not in p.products(root)
    array.write_bytes(b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):p.verify_predictions(root)


def test_explicit_code_allowlist_and_both_commit_bindings(tmp_path,monkeypatch):
    monkeypatch.setattr(p.guard,'ROOT',tmp_path)
    for name in p.PRODUCTION:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
    calls=[]
    def verify(commit,names):
        calls.append((commit,list(names)));return {name:HASH for name in names}
    monkeypatch.setattr(p.guard,'verify_commit',verify)
    binding=p.code_binding(CODE,SPEC)
    assert calls==[(CODE,[*p.PRODUCTION,p.SPECIFICATION]),(SPEC,[p.SPECIFICATION])]
    assert binding['specification_sha256']==HASH
    (tmp_path/'offline_evaluation/analysis/goal010/unbound.py').write_text('# extra\n')
    with pytest.raises(ValueError,match='allowlist'):p.code_binding(CODE,SPEC)
