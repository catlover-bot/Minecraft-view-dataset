"""Independent synthetic preflight checks; never create canonical runs or hooks."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from offline_evaluation.analysis.goal013 import run_preflight as p


RULE = b'{"fixture_selection":"fixed_before_scores"}\n'


def blocked_audit():
    return {
        'status': p.BLOCKED,
        'computation_status': 'COMPUTATION_NOT_PERFORMED',
        'validated_offline_scorer_count': 0,
        'selected_candidate': None,
        'scored_description_count': 0,
        'target_description_count': 180,
        'scored_image_text_pair_count': 0,
        'score_validity': 'NOT_ASSESSABLE_NO_SCORER',
        'actual_correlation_n': None,
        'paired_denominators': None,
        'selection_rule': {'sha256': hashlib.sha256(RULE).hexdigest()},
        'candidates': [{'eligibility': 'INELIGIBLE', 'checkpoint_id': None}],
        'actions': {
            name: 0 for name in (
                'research_generation_api_calls', 'model_lookup_api_calls',
                'token_count_api_calls', 'judge_api_calls', 'inference_calls',
                'human_experiments', 'secret_or_authentication_queries',
                'environment_variable_value_reads', 'downloads',
                'package_installs', 'existing_venv_mutations',
                'existing_cache_mutations',
            )
        },
    }


def test_accepts_only_unscored_no_candidate_audit():
    audit = blocked_audit()
    before = deepcopy(audit)
    p.require_blocked_scorer(audit, RULE)
    assert audit == before


@pytest.mark.parametrize('field,value', [
    ('status', 'PASS'),
    ('validated_offline_scorer_count', 1),
    ('selected_candidate', {'checkpoint_id': 'fixture/model'}),
    ('scored_description_count', 1),
    ('scored_image_text_pair_count', 1),
    ('computation_status', 'SCORED'),
    ('actual_correlation_n', 1),
    ('paired_denominators', {'openai': 1}),
])
def test_rejects_candidate_score_or_association_contamination(field, value):
    audit = blocked_audit()
    audit[field] = value
    with pytest.raises(ValueError):
        p.require_blocked_scorer(audit, RULE)


@pytest.mark.parametrize('action', [
    'inference_calls', 'downloads', 'package_installs',
    'research_generation_api_calls', 'secret_or_authentication_queries',
])
def test_rejects_nonzero_execution_claims_in_blocked_audit(action):
    audit = blocked_audit()
    audit['actions'][action] = 1
    with pytest.raises(ValueError):
        p.require_blocked_scorer(audit, RULE)


def test_rejects_rule_bytes_changed_after_availability_audit():
    with pytest.raises(ValueError, match='rule'):
        p.require_blocked_scorer(blocked_audit(), RULE + b' ')


@pytest.fixture
def committed_fixture(tmp_path, monkeypatch):
    commit = 'a' * 40
    names = [
        'offline_evaluation/analysis/goal013/source_binding.py',
        'offline_evaluation/analysis/goal013/run_preflight.py',
        'reports/server_only/goal013/SCORING_PROTOCOL.md',
        'reports/server_only/goal013/MODEL_SELECTION_RULE.json',
        'reports/server_only/goal013/OFFLINE_SCORER_AUDIT.json',
    ]
    blobs = {name: ('synthetic committed fixture: ' + name).encode() for name in names}
    for name, raw in blobs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    state = {'head': commit, 'names': names}

    def git(argv, **kwargs):
        assert kwargs['cwd'] == tmp_path
        if argv == ['git', 'rev-parse', 'HEAD']:
            return state['head'].encode()
        if argv[:3] == ['git', 'ls-tree', '-r']:
            return ('\n'.join(state['names']) + '\n').encode()
        assert argv[:2] == ['git', 'show']
        return blobs[argv[2].split(':', 1)[1]]

    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p.subprocess, 'check_output', git)
    return tmp_path, commit, blobs, state


def test_committed_protocol_and_current_bytes_match(committed_fixture):
    root, commit, blobs, _ = committed_fixture
    descriptors = p.verify_preflight_commit(commit)
    assert set(descriptors) == set(blobs)
    assert all(descriptors[name]['sha256'] == hashlib.sha256(raw).hexdigest()
               for name, raw in blobs.items())


@pytest.mark.parametrize('change', ['protocol_bytes', 'protocol_absent', 'head', 'short_commit'])
def test_rejects_uncommitted_or_changed_protocol(committed_fixture, change):
    root, commit, _, state = committed_fixture
    protocol = 'reports/server_only/goal013/SCORING_PROTOCOL.md'
    if change == 'protocol_bytes':
        (root / protocol).write_text('changed after commit')
    elif change == 'protocol_absent':
        state['names'] = [name for name in state['names'] if name != protocol]
    elif change == 'head':
        state['head'] = 'b' * 40
    else:
        commit = commit[:8]
    with pytest.raises(ValueError):
        p.verify_preflight_commit(commit)


@pytest.mark.parametrize('path', [
    'relative',
    '/tmp/goal013-20261017T000000Z',
    '/root/minecraft-runs/goal011-20261017T000000Z',
    '/root/minecraft-runs/goal013-20261017T000000Z/nested',
    '/root/minecraft-runs/goal013-20261017T000000Z/../goal013-20261017T000001Z',
    '/root/minecraft-runs/goal013-existing',
])
def test_invalid_output_roots_are_rejected_before_mkdir(path, monkeypatch):
    def forbidden_mkdir(*args, **kwargs):
        pytest.fail('invalid path reached mkdir')
    monkeypatch.setattr(Path, 'mkdir', forbidden_mkdir)
    with pytest.raises(ValueError):
        p.prepare_output(Path(path))


def test_existing_canonical_run_is_never_reused(monkeypatch):
    def occupied(path, **kwargs):
        assert kwargs == {'exist_ok': False}
        raise FileExistsError('synthetic existing run; no real directory created')
    monkeypatch.setattr(Path, 'mkdir', occupied)
    with pytest.raises(FileExistsError):
        p.prepare_output(Path('/root/minecraft-runs/goal013-20261017T000000Z'))


@pytest.fixture
def captured_write_guard(tmp_path, monkeypatch):
    destination = tmp_path / 'new-run'
    destination.mkdir()
    hooks = []
    monkeypatch.setattr(p.sys, 'addaudithook', hooks.append)
    p.restrict_writes(destination)
    assert len(hooks) == 1
    return destination, hooks[0]


def test_write_guard_allows_run_output_and_outside_reads(captured_write_guard):
    destination, hook = captured_write_guard
    hook('open', (str(destination / 'receipt.json'), 'x', os.O_WRONLY | os.O_CREAT))
    hook('open', (str(destination.parent / 'source.json'), 'r', os.O_RDONLY))


@pytest.mark.parametrize('kind', ['outside', 'traversal', 'symlink'])
def test_write_guard_rejects_external_writes(captured_write_guard, kind):
    destination, hook = captured_write_guard
    path = destination.parent / 'protected.json'
    if kind == 'traversal':
        path = destination / '..' / 'protected.json'
    elif kind == 'symlink':
        link = destination / 'escape'
        link.symlink_to(destination.parent, target_is_directory=True)
        path = link / 'protected.json'
    with pytest.raises(ValueError):
        hook('open', (str(path), 'w', os.O_WRONLY | os.O_CREAT))


@pytest.mark.parametrize('event,args', [
    ('os.mkdir', ('protected-dir', 0o700, -1)),
    ('os.remove', ('protected-file', -1)),
    ('os.rename', ('inside', 'protected-file', -1, -1)),
])
def test_write_guard_rejects_external_mutations(captured_write_guard, event, args):
    destination, hook = captured_write_guard
    paths = tuple(str(destination / part) if part == 'inside' else
                  str(destination.parent / part) if isinstance(part, str) else part
                  for part in args)
    with pytest.raises(ValueError):
        hook(event, paths)


@pytest.fixture
def main_fixture(tmp_path, monkeypatch):
    reports = tmp_path / 'reports'
    reports.mkdir()
    (reports / 'OFFLINE_SCORER_AUDIT.json').write_text(json.dumps(blocked_audit()))
    (reports / 'MODEL_SELECTION_RULE.json').write_bytes(RULE)
    destination = tmp_path / 'new-run'
    state = {'locked': False, 'exit_failure': False, 'events': []}

    def prepare(_):
        destination.mkdir(exist_ok=False)
        return destination

    @contextmanager
    def locked():
        state['locked'] = True
        state['events'].append('lock_enter')
        try:
            yield ['fixture-lock']
            state['events'].append('lock_exit_check')
            assert not (destination / 'PREFLIGHT_RECEIPT.json').exists()
            if state['exit_failure']:
                raise ValueError('synthetic exit integrity failure')
        finally:
            state['locked'] = False

    def audit_sources():
        assert state['locked']
        state['events'].append('audit_sources')
        return {'rows': [{'score': None, 'score_status': 'COMPUTATION_NOT_PERFORMED'} for _ in range(180)],
                'receipt': {'source_inventory': {}, 'independent_buildings': 60}}

    def verify(_):
        assert state['locked']
        state['events'].append('verify_inventory')

    monkeypatch.setattr(p, 'REPORTS', reports)
    monkeypatch.setattr(p, 'verify_preflight_commit', lambda _: {'fixture': {'sha256': 'a' * 64}})
    monkeypatch.setattr(p, 'prepare_output', prepare)
    monkeypatch.setattr(p, 'restrict_writes', lambda _: None)
    monkeypatch.setattr(p.resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(p.signal, 'alarm', lambda *args: None)
    monkeypatch.setattr(p, 'locked_sources', locked)
    monkeypatch.setattr(p.sources, 'audit_sources', audit_sources)
    monkeypatch.setattr(p.sources, 'write_outputs', lambda *args: {})
    monkeypatch.setattr(p.sources, 'verify_inventory', verify)
    monkeypatch.setattr(p.sys, 'argv', ['fixture', '--output-root', str(destination), '--code-commit', 'a' * 40])
    return destination, state


def test_receipt_is_written_only_after_exit_integrity_check(main_fixture):
    destination, state = main_fixture
    assert p.main() == 0
    receipt = json.loads((destination / 'PREFLIGHT_RECEIPT.json').read_text())
    assert state['events'] == ['lock_enter', 'audit_sources', 'verify_inventory', 'lock_exit_check']
    assert receipt['status'] == p.BLOCKED and receipt['source_binding_status'] == 'PASS'
    assert receipt['scored_descriptions'] == 0 and receipt['actual_association_N'] is None
    assert receipt['phase1_executed'] is False and receipt['phase2_executed'] is False


def test_exit_integrity_failure_never_leaves_a_success_receipt(main_fixture):
    destination, state = main_fixture
    state['exit_failure'] = True
    with pytest.raises(ValueError, match='exit integrity'):
        p.main()
    assert not (destination / 'PREFLIGHT_RECEIPT.json').exists()
    assert json.loads((destination / 'FAILED.json').read_text())['status'] == 'SOURCE_BINDING_FAILED'
