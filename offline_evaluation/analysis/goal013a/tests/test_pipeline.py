"""Pipeline boundaries with fake sessions and temporary synthetic sources only.

No model is loaded; no saved research source/outcome is opened; network events
are synthetic audit events rather than actual communication attempts.
"""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from offline_evaluation.analysis.goal013a import run_pipeline as p
from offline_evaluation.analysis.goal013a import scorer, reference_parity, integrity
from offline_evaluation.analysis.goal013 import source_binding


def test_offline_guard_blocks_network_without_making_connections(tmp_path):
    with p.offline_guard([tmp_path]) as state:
        for event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.sendto'):
            with pytest.raises(RuntimeError, match='network prohibited'):
                sys.audit(event, None)
        assert state['network_attempts_blocked'] == 4
    assert state['active'] is False


def test_offline_guard_allows_only_new_area_writes_and_refuses_symlink_escape(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    original = tmp_path / 'old.txt'
    original.write_bytes(b'original')
    (output / 'escape').symlink_to(original)
    with p.offline_guard([output]) as state:
        (output / 'new.json').write_text('{}')
        for target in (original, output / 'escape'):
            with pytest.raises(RuntimeError, match='write outside'):
                target.write_text('changed')
        with pytest.raises(RuntimeError, match='write outside'):
            original.rename(output / 'moved.txt')
        assert original.read_bytes() == b'original'
        assert state['source_writes_blocked'] == 3
    assert original.read_bytes() == b'original'


@pytest.mark.parametrize('event', ['os.system', 'os.posix_spawn', 'os.exec', 'os.fork', 'os.forkpty'])
def test_shell_and_spawn_bypasses_are_refused_without_executing(event, tmp_path):
    with p.offline_guard([tmp_path]):
        with pytest.raises(RuntimeError, match='process execution'):
            sys.audit(event, 'synthetic-only')


def test_git_subprocess_allowlist_is_read_only(tmp_path):
    with p.offline_guard([tmp_path]):
        for command in ('show', 'rev-parse', 'ls-files'):
            sys.audit('subprocess.Popen', 'git', ['git', command], None, None)
        for args in (['git', 'push'], ['git', 'reset'], ['python', '-V'], 'git show HEAD'):
            with pytest.raises(ValueError, match='read-only Git'):
                sys.audit('subprocess.Popen', 'fake', args, None, None)


def test_resource_signals_abort_and_preserve_fixed_cleanup_margin(monkeypatch, tmp_path):
    limits, handlers, alarms = [], {}, []
    monkeypatch.setattr(p.time, 'process_time', lambda: 12.25)
    monkeypatch.setattr(p.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(p.resource, 'setrlimit', lambda kind, value: limits.append((kind, value)))
    monkeypatch.setattr(p.signal, 'signal', lambda sig, handler: handlers.__setitem__(sig, handler))
    monkeypatch.setattr(p.signal, 'alarm', lambda value: alarms.append(value))
    p.Budget(tmp_path, tmp_path, 6 * p.GIB)
    assert (p.resource.RLIMIT_AS, (6 * p.GIB, 6 * p.GIB)) in limits
    assert (p.resource.RLIMIT_CPU, (3613, 3733)) in limits
    assert alarms == [7200]
    stop = handlers[p.signal.SIGXCPU]
    with pytest.raises(RuntimeError, match='resource limit'):
        stop(p.signal.SIGXCPU, None)
    assert alarms[-1] == 0
    assert handlers[p.signal.SIGXCPU] == p.signal.SIG_IGN


@pytest.mark.parametrize('field,value', [('cpu_seconds', 3600), ('elapsed_seconds', 7200), ('max_rss_kib', 6 * 1024**2 + 1)])
def test_time_and_rss_caps_are_not_automatically_raised(field, value, tmp_path):
    budget = object.__new__(p.Budget)
    budget.memory_limit = 6 * p.GIB
    state = {'cpu_seconds': 1., 'elapsed_seconds': 1., 'max_rss_kib': 1}
    state[field] = value
    budget.snapshot = lambda: state
    with pytest.raises(ValueError, match='budget|memory'):
        budget.check()


@pytest.mark.parametrize('size,free,passes', [(8 * p.GIB, 10 * p.GIB, True), (8 * p.GIB + 1, 20 * p.GIB, False), (1, 10 * p.GIB - 1, False)])
def test_disk_limits_without_creating_large_files(monkeypatch, size, free, passes):
    fake_file = SimpleNamespace(is_file=lambda: True, is_symlink=lambda: False,
                                stat=lambda: SimpleNamespace(st_size=size))
    budget = object.__new__(p.Budget)
    budget.tools_root = SimpleNamespace(rglob=lambda pattern: [fake_file])
    budget.run_root = SimpleNamespace(rglob=lambda pattern: [])
    budget.memory_limit = 6 * p.GIB
    budget.snapshot = lambda: {'cpu_seconds': 1., 'elapsed_seconds': 1., 'max_rss_kib': 1}
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda root: SimpleNamespace(free=free))
    if passes:
        assert budget.check(disk=True)['new_work_area_bytes'] == size
    else:
        with pytest.raises(ValueError, match='GiB'):
            budget.check(disk=True)


def test_pilot_projection_fixed_multiplier_and_strict_time_cap():
    pilot = {'cpu_seconds': 300., 'elapsed_seconds': 600.}
    assert p.projection(pilot, {'cpu_seconds': 100., 'elapsed_seconds': 100.})['status'] == 'PASS'
    failed = p.projection(pilot, {'cpu_seconds': 600., 'elapsed_seconds': 1200.})
    assert failed['status'] == 'RESOURCE_PROJECTION_FAILED'
    assert failed['safety_multiplier'] == 2 and failed['remaining_buildings'] == 50


def test_public_score_copy_omits_token_ids_without_mutating_canonical_rows():
    rows = [{'token_diagnostics': {'token_ids': [1, 2, 3], 'token_ids_sha256': 'a' * 64}, 'score': None}]
    copy = p.public_rows(rows)
    assert 'token_ids' not in copy[0]['token_diagnostics']
    assert rows[0]['token_diagnostics']['token_ids'] == [1, 2, 3]
    assert copy[0]['token_diagnostics']['token_ids_sha256'] == 'a' * 64


def commit_fixture(monkeypatch, tmp_path):
    code_commit, spec_commit = '1' * 40, '2' * 40
    names = {code_commit: p.CODE + 'fake.py', spec_commit: p.REPORT + 'config.json'}
    content = {names[code_commit]: b'# synthetic code\n', names[spec_commit]: b'{}\n'}
    for name, raw in content.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    state = {'branch': 'research/server-only-20261013', 'status': '', 'ancestry': []}
    def git(*args):
        if args[0] == 'branch':
            return state['branch']
        if args[0] == 'ls-tree':
            return names[args[3]]
        if args[0] == 'status':
            return state['status']
        raise AssertionError(args)
    monkeypatch.setattr(p, 'git', git)
    def show(args, cwd):
        assert args[:2] == ['git', 'show'] and cwd == tmp_path
        return content[args[2].split(':', 1)[1]]
    monkeypatch.setattr(p.subprocess, 'check_output', show)
    monkeypatch.setattr(p.subprocess, 'run', lambda args, **kwargs: state['ancestry'].append(args))
    return code_commit, spec_commit, tmp_path / names[spec_commit], state


def test_commit_gate_checks_ancestry_and_exact_committed_bytes(monkeypatch, tmp_path):
    code, spec, config, state = commit_fixture(monkeypatch, tmp_path)
    result = p.verify_commits(code, spec, config)
    assert len(result) == 2
    assert state['ancestry'] == [['git', 'merge-base', '--is-ancestor', code, spec],
                                 ['git', 'merge-base', '--is-ancestor', spec, 'HEAD']]


@pytest.mark.parametrize('mutation', ['branch', 'dirty_code', 'changed_code', 'changed_spec', 'same_commit', 'uncommitted_config'])
def test_commit_gate_rejects_changed_or_uncommitted_method(monkeypatch, tmp_path, mutation):
    code, spec, config, state = commit_fixture(monkeypatch, tmp_path)
    if mutation == 'branch':
        state['branch'] = 'main'
    elif mutation == 'dirty_code':
        state['status'] = '?? added_code.py'
    elif mutation == 'changed_code':
        (tmp_path / p.CODE / 'fake.py').write_bytes(b'changed')
    elif mutation == 'changed_spec':
        config.write_bytes(b'changed')
    elif mutation == 'same_commit':
        spec = code
    else:
        config = tmp_path / 'uncommitted.json'
    with pytest.raises(ValueError):
        p.verify_commits(code, spec, config)


class FakeBudget:
    def __init__(self, *args):
        self.checks = []
    def snapshot(self):
        return {'cpu_seconds': 1., 'elapsed_seconds': 1., 'max_rss_kib': 1}
    def check(self, disk=False):
        self.checks.append(disk)
        return self.snapshot()


def score_fixture(tmp_path, monkeypatch, *, parity_status='PASS', bad_gate=False, fail_text_at=None):
    rows = []
    for scene_index in range(60):
        scene = f'synthetic_scene_{scene_index:03}'
        images = [{'path': f'/synthetic/{scene}/view{v}.png', 'sha256': str(v) * 64, 'bytes': 1} for v in range(6)]
        for provider in ('anthropic', 'gemini', 'openai'):
            empty = provider == 'gemini' and scene_index in (54, 59)
            rows.append({'scene_id': scene, 'description_provider': provider, 'summary_sha256': '3' * 64,
                         'summary_status': 'MISSING_EMPTY_SUMMARY' if empty else 'VALID_ORIGINAL_SUMMARY',
                         'images_ordered': images})
    path = tmp_path / 'binding.json'
    path.write_text(json.dumps(rows))
    config_path = tmp_path / 'config.json'
    config_path.write_text('{}')
    config = {'planned_scene_ids': sorted({r['scene_id'] for r in rows}), 'score_id': scorer.SCORE_ID,
              'tolerances': scorer.TOLERANCES, 'transform': scorer.TRANSFORM, '_config_path': str(config_path),
              'parity_input': {k: rows[0][k] for k in ('scene_id', 'description_provider', 'summary_sha256')}}
    verified_paths, events = set(), []
    def load_record(row):
        paths = [image['path'] for image in row['images_ordered']]
        verified_paths.update(paths)
        events.append(('source_verified', row['scene_id'], row['description_provider']))
        return {'summary': 'synthetic summary', 'images_ordered': paths}
    monkeypatch.setattr(source_binding, 'load_scoring_record', load_record)
    class Session:
        image_cache = {}
        text_cache = {}
        cache_counts = {'image_encoded': 0, 'text_encoded': 0}
        tokenizer_identity = '4' * 64
        text_calls = 0
        def encode_images(self, paths, batch_size):
            assert batch_size <= 8 and set(paths) <= verified_paths
            events.append(('image_encoding', tuple(paths)))
            self.cache_counts['image_encoded'] += len(paths)
            return np.tile(np.array([[1., 0.]], dtype=np.float32), (6, 1))
        def encode_texts(self, summaries, batch_size):
            assert summaries == ['synthetic summary'] and batch_size <= 8
            self.text_calls += 1
            if self.text_calls == fail_text_at:
                raise RuntimeError('synthetic inference interruption')
            self.cache_counts['text_encoded'] += 1
            return np.array([[1., 0.]], dtype=np.float32)
        def token_diagnostics(self, summary):
            if summary is None:
                return {'token_coverage_status': 'MISSING_SUMMARY', 'summary_token_count_untruncated': None,
                        'token_ids': None}
            return {'token_coverage_status': 'VERIFIED', 'summary_token_count_untruncated': 10,
                    'summary_dropped_fraction': .7 if bad_gate else 0., 'truncated': bad_gate, 'token_ids': [1, 2, 3]}
        def crop_preview(self, path):
            assert path in verified_paths
            preview = SimpleNamespace(save=lambda stream, format: stream.write(b'synthetic preview'))
            return preview, {'contains_complete_building': 'NOT_VERIFIED'}
    session = Session()
    monkeypatch.setattr(scorer, 'load_model', lambda config: session)
    def parity(session_arg, images, summaries):
        assert session_arg is session and set(images) <= verified_paths and len(summaries) == 6
        events.append(('parity',))
        return {'status': parity_status, 'fixture_only': True}
    monkeypatch.setattr(reference_parity, 'run_reference_parity', parity)
    output = tmp_path / 'score'
    output.mkdir()
    return config, output, {'canonical_products': {'SOURCE_BINDING_180.json': str(path)}}, FakeBudget(), events


def test_complete_synthetic_pipeline_retains_missing_rows_and_uses_verified_images_first(tmp_path, monkeypatch):
    config, out, binding, budget, events = score_fixture(tmp_path, monkeypatch)
    freeze = p.score_phase(config, out, '1' * 40, '2' * 40, binding, budget)
    summary = json.loads((out / 'SCORING_SUMMARY.json').read_text())
    assert summary['computed_rows'] == 178 and summary['computed_image_text_pairs'] == 1068
    assert summary['planned_rows'] == 180 and summary['planned_image_references'] == 1080
    assert summary['unique_images'] == 360 and freeze['phase2_allowed'] is True
    assert len(list(out.glob('row_*.json'))) == 180
    assert len(list(out.glob('*_fixed_first_crop.png'))) == 3
    assert sum(budget.checks) >= 60
    public = json.loads(Path(freeze['score_artifact']['path']).read_text())
    canonical = json.loads(Path(freeze['canonical_token_score_artifact']['path']).read_text())
    assert all('token_ids' not in row['token_diagnostics'] for row in public)
    assert canonical[0]['token_diagnostics']['token_ids'] == [1, 2, 3]
    missing = [row for row in public if row['score_status'] == 'MISSING_SUMMARY']
    assert len(missing) == 2 and all(row['score'] is None for row in missing)
    assert not (out / 'SCORE_FREEZE.json').exists()  # main must verify old-source exit first.
    assert sum(event[0] == 'image_encoding' for event in events) == 60


def test_failed_reference_parity_preserves_receipt_and_prevents_dataset_scoring(tmp_path, monkeypatch):
    config, out, binding, budget, events = score_fixture(tmp_path, monkeypatch, parity_status='REFERENCE_PARITY_FAILED')
    with pytest.raises(ValueError, match='parity gate'):
        p.score_phase(config, out, '1' * 40, '2' * 40, binding, budget)
    assert (out / 'REFERENCE_PARITY.json').is_file()
    assert not list(out.glob('row_*.json'))
    assert not any(event[0] == 'image_encoding' for event in events)


def test_failed_representativeness_retains_scores_without_authorizing_association(tmp_path, monkeypatch):
    config, out, binding, budget, _ = score_fixture(tmp_path, monkeypatch, bad_gate=True)
    frozen = p.score_phase(config, out, '1' * 40, '2' * 40, binding, budget)
    assert frozen['phase2_allowed'] is False
    assert (out / 'SCORES_180.json').is_file()
    gate = json.loads((out / 'TOKEN_REPRESENTATIVENESS.json').read_text())
    assert gate['status'] == 'INSUFFICIENT_MEASURE_VALIDITY'
    assert not (out / 'OUTCOME_BINDING.json').exists()


def test_mid_scene_interruption_keeps_each_completed_row(tmp_path, monkeypatch):
    config, out, binding, budget, _ = score_fixture(tmp_path, monkeypatch, fail_text_at=2)
    with pytest.raises(RuntimeError, match='synthetic inference'):
        p.score_phase(config, out, '1' * 40, '2' * 40, binding, budget)
    partial = list(out.glob('row_*.json'))
    assert len(partial) == 1
    assert json.loads(partial[0].read_text())['score_status'] == 'COMPUTATION_VALID'
    assert not (out / 'SCORES_180.json').exists()


def main_fixture(tmp_path, monkeypatch, *, stage='score', end_integrity_failure=False, stage_failure=False):
    tools, run = tmp_path / 'tools', tmp_path / 'run'
    tools.mkdir(); run.mkdir()
    for path in (tools / 'tmp', tools / 'cache', tools / 'venv'):
        path.mkdir()
    config = {'tools_root': str(tools), 'run_root': str(run), 'memory_limit_bytes': 6 * p.GIB,
              'installed_distributions': {}}
    config_path = tmp_path / 'main_config.json'
    config_path.write_text(json.dumps(config))
    events = []
    monkeypatch.setattr(p.sys, 'prefix', str(tools / 'venv'))
    monkeypatch.setattr(p.sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(p.sys, 'argv', ['fixture', stage, '--config', str(config_path),
                                      '--code-commit', '1' * 40, '--spec-commit', '2' * 40])
    for env in p.THREAD_ENV:
        monkeypatch.setenv(env, '1')
    for env, path in {'TMPDIR': tools / 'tmp', 'XDG_CACHE_HOME': tools / 'cache',
                      'TORCH_HOME': tools / 'cache/torch', 'PIP_CACHE_DIR': tools / 'cache/pip'}.items():
        monkeypatch.setenv(env, str(path))
    # Only production root geometry is bypassed: all test files stay in tmp.
    real_require = p.require
    def fixture_require(ok, message):
        if message not in ('new dedicated tools root required', 'new dedicated run root required'):
            real_require(ok, message)
    monkeypatch.setattr(p, 'require', fixture_require)
    monkeypatch.setattr(p, 'verify_commits', lambda *args: events.append('commits_verified'))
    monkeypatch.setattr(p, 'Budget', FakeBudget)
    monkeypatch.setattr(p.signal, 'alarm', lambda n: None)
    monkeypatch.setattr(integrity, 'resource_snapshot', lambda: {'recommended_process_memory_limit_bytes': 6 * p.GIB})
    @contextmanager
    def protected(path):
        events.append('old_sources_verified_before')
        try:
            yield 2
        finally:
            events.append('old_sources_verified_after')
            assert not (run / stage / 'SCORE_FREEZE.json').exists()
            if end_integrity_failure:
                raise RuntimeError('synthetic old-source exit mismatch')
    monkeypatch.setattr(integrity, 'protected_sources', protected)
    monkeypatch.setattr(integrity, 'verify_goal013', lambda: events.append('source_binding_verified') or {})
    def score(config, out, *args):
        events.append('scoring')
        p.write_json(out / 'row_001.json', {'score_status': 'COMPUTATION_VALID', 'score': .1})
        if stage_failure:
            raise RuntimeError('synthetic resource interruption')
        return {'phase2_allowed': False, 'representativeness_artifact': {'fixture': True}}
    monkeypatch.setattr(p, 'score_phase', score)
    return run / stage, events


def test_main_freezes_only_after_exit_integrity_and_marks_phase2_not_run(tmp_path, monkeypatch):
    out, events = main_fixture(tmp_path, monkeypatch)
    p.main()
    assert events == ['commits_verified', 'old_sources_verified_before', 'source_binding_verified', 'scoring', 'old_sources_verified_after']
    assert (out / 'SCORE_FREEZE.json').is_file() and (out / 'EXECUTION_RECEIPT.json').is_file()
    stopped = json.loads((out / 'PHASE2_NOT_RUN.json').read_text())
    assert stopped['association_N'] is None and stopped['bootstrap_draws_executed'] == 0


@pytest.mark.parametrize('end_integrity_failure,stage_failure', [(True, False), (False, True)])
def test_main_failure_never_freezes_and_reports_retained_rows(tmp_path, monkeypatch, end_integrity_failure, stage_failure):
    out, _ = main_fixture(tmp_path, monkeypatch, end_integrity_failure=end_integrity_failure, stage_failure=stage_failure)
    with pytest.raises(RuntimeError, match='synthetic'):
        p.main()
    receipt = json.loads((out / 'STOP_RECEIPT.json').read_text())
    assert receipt['retained_planned_rows'] == receipt['retained_computed_rows'] == 1
    assert receipt['remaining_planned_rows'] == 179
    assert (out / 'row_001.json').is_file() and not (out / 'SCORE_FREEZE.json').exists()


def test_association_stage_cannot_ignore_failed_freeze_gate(tmp_path, monkeypatch):
    from offline_evaluation.analysis.goal013a import outcome_binding, associations
    out, _ = main_fixture(tmp_path, monkeypatch, stage='associate')
    def refuse(*args, **kwargs):
        raise ValueError('synthetic freeze rejected before outcomes')
    monkeypatch.setattr(outcome_binding, 'load_frozen_outcomes', refuse)
    monkeypatch.setattr(associations, 'analyze', lambda *args, **kwargs: pytest.fail('arithmetic entered after rejected freeze'))
    with pytest.raises(ValueError, match='freeze rejected'):
        p.main()
    assert (out / 'STOP_RECEIPT.json').is_file()
    assert not (out / 'OUTCOME_BINDING.json').exists() and not (out / 'ASSOCIATIONS_FULL.json').exists()


@pytest.mark.parametrize('variable', [*p.THREAD_ENV, 'TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'PIP_CACHE_DIR'])
def test_environment_mismatch_stops_before_output_or_scoring(tmp_path, monkeypatch, variable):
    out, events = main_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(variable, 'incorrect')
    with pytest.raises(ValueError, match='environment'):
        p.main()
    assert not out.exists() and 'scoring' not in events
