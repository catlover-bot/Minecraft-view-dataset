"""Boundary fixtures for the new entry point; no real model or experiments."""
import json
from pathlib import Path

import pytest

from offline_evaluation.analysis.goal014 import run_pipeline as p


def test_bound_file_detects_tampering(tmp_path):
    source = tmp_path/'source.json'; source.write_text('{"value": 1}')
    bound = p.descriptor(source)
    assert p.read_bound(bound) == {'value': 1}
    source.write_text('{"value": 2}')
    with pytest.raises(ValueError, match='bound file changed'):
        p.read_bound(bound)


def test_exclusive_output_keeps_original(tmp_path):
    out = tmp_path/'out.json'; p.write_json(out, {'value': 1})
    before = out.read_bytes()
    with pytest.raises(ValueError, match='exclusive'):
        p.write_json(out, {'value': 2})
    assert out.read_bytes() == before


def test_guard_keeps_old_score_and_statistics_unchanged(tmp_path):
    new = tmp_path/'new'; new.mkdir()
    old_score = tmp_path/'SCORES.json'; old_stats = tmp_path/'STATISTICS.json'
    old_score.write_text('score'); old_stats.write_text('statistics')
    with p.offline_guard([new]) as state:
        for old in (old_score, old_stats):
            with pytest.raises(RuntimeError, match='outside new dedicated'):
                old.write_text('overwrite')
        (new/'allowed.json').write_text('{}')
    assert old_score.read_text() == 'score' and old_stats.read_text() == 'statistics'
    assert state['source_writes_blocked'] == 2


def test_environment_stays_in_new_run(tmp_path, monkeypatch):
    names = ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','TORCH_HOME','PIP_CACHE_DIR','HF_HOME','MPLCONFIGDIR',
             'PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
    for name in names: monkeypatch.setenv(name, 'original')
    p.configure_environment(tmp_path)
    import os
    for name in ('TMPDIR','TMP','TEMP'): assert os.environ[name] == str(tmp_path/'tmp')
    for name in ('XDG_CACHE_HOME','TORCH_HOME','PIP_CACHE_DIR','HF_HOME','MPLCONFIGDIR'):
        assert os.environ[name] == str(tmp_path/'cache')
    assert os.environ['OPENBLAS_NUM_THREADS'] == '1'


def test_budget_refuses_elapsed_goal(tmp_path, monkeypatch):
    monkeypatch.setattr(p.integrity, 'resource_snapshot', lambda: {'recommended_process_memory_limit_bytes': 6*p.GIB})
    config = {'limits': {'rss_max_bytes':6*p.GIB,'cpu_seconds':1200,'wall_seconds':1800},
              'goal_start_epoch':0,'pre_execution_cpu_reserve_seconds':180,'closure_cpu_reserve_seconds':120}
    with pytest.raises(ValueError, match='budget exhausted'):
        p.Budget(config,tmp_path)
