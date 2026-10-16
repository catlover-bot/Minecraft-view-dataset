from pathlib import Path
from types import SimpleNamespace

import pytest

from offline_evaluation.analysis.goal007 import cli


def test_invalid_output_cannot_create_or_touch_sources(tmp_path):
    target=tmp_path/'outside_research_directory'
    with pytest.raises(cli.GenerationError,match='new_research_output'):
        cli.run(SimpleNamespace(destination=str(target)))
    assert not target.exists()


def test_failed_checkpoint_gate_stops_before_output_or_statistics(monkeypatch):
    target=Path('/root/minecraft-runs/goal007-fixture-must-never-be-created')
    assert not target.exists()
    monkeypatch.setattr(cli,'verify_code_commit',lambda _: {})
    def failed(_): raise cli.audit.AuditError('checkpoint_corrupt_fixture')
    monkeypatch.setattr(cli.audit,'verify_gate',failed)
    def forbidden(*a,**k): raise AssertionError('statistics must not be called')
    monkeypatch.setattr(cli.statistics,'analyze_slots',forbidden)
    with pytest.raises(cli.audit.AuditError,match='checkpoint_corrupt'):
        cli.run(SimpleNamespace(destination=str(target),code_commit='a'*40,audit_gate='fixture'))
    assert not target.exists()


def test_explicit_implementation_commit_required():
    with pytest.raises(cli.GenerationError,match='explicit_code_commit'):
        cli.verify_code_commit('HEAD')


def test_failure_layers_never_merge_attempts_and_selected_slots():
    rows=[{'provider':'openai','condition':'O','status':'failed','failure_class':'operational_unknown','failure_kind':'auth_failure'},
          {'provider':'openai','condition':'J','status':'failed','failure_class':'operational_unknown','failure_kind':'http_failure'},
          {'provider':'anthropic','condition':'E','status':'failed','failure_class':'model_output_failure','failure_kind':'truncation'}]
    selected=[dict(rows[1],status='operational_unmeasurable'),rows[2]]
    taxonomy,model,operational=cli.failure_tables({'attempts':rows,'rows':selected})
    assert len(model)==1 and len(operational)==2
    assert sum(r['count'] for r in taxonomy if r['record_type']=='attempt')==3
    assert sum(r['count'] for r in taxonomy if r['record_type']=='selected_slot')==2


def test_completion_writes_manifest_using_real_path_json_reader(tmp_path, monkeypatch):
    """Exercise the post-render boundary, including the actual frozen read_json.

    Scientific work is mocked; the fixture only verifies CLI orchestration and
    fresh-file finalization. All writes stay inside pytest's temporary directory.
    """
    import json
    from contextlib import nullcontext

    actual_path = Path
    research_root = tmp_path / 'research'; research_root.mkdir()
    destination = research_root / 'new-analysis'
    # Map only the production output-root policy constant into this fixture.
    monkeypatch.setattr(cli, 'Path', lambda value: research_root if str(value) == '/root/minecraft-runs' else actual_path(value))
    source = tmp_path / 'immutable-source.json'; source.write_text('{"fixture":"preserve"}\n')
    original_source_bytes = source.read_bytes()
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'files': {str(source): {'sha256': cli.sha256_file(source), 'bytes': source.stat().st_size}}, 'trees': {}}))
    gate_path = tmp_path / 'audit_gate.json'
    gate = {'status': 'PASS', 'sources': {}, 'github_checkpoints': {}, 'parent160_checkpoint': {},
            'manifest_path': str(source), 'manifest_sha256': cli.sha256_file(source),
            'inventory_path': str(inventory), 'immutable_file_count': 1}
    gate_path.write_text(json.dumps(gate))
    rq2_path = tmp_path / 'rq2.json'; rq2_path.write_text(json.dumps({'status': 'NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES'}))
    hashes = {'fixture_implementation.py': 'fixture-hash'}
    monkeypatch.setattr(cli, 'verify_code_commit', lambda _: hashes)
    monkeypatch.setattr(cli, 'implementation_hashes', lambda: hashes)
    gate_calls = []
    def verify_gate(value):
        gate_calls.append(value)
        assert source.read_bytes() == original_source_bytes
        return gate
    monkeypatch.setattr(cli.audit, 'verify_gate', verify_gate)
    monkeypatch.setattr(cli.audit, 'source_locks', lambda _: nullcontext())
    monkeypatch.setattr(cli.audit, 'process_audit', lambda: {})
    monkeypatch.setattr(cli.integration, 'load_sources', lambda _: ({'identity_hash': 'fixture-manifest'}, {}))
    monkeypatch.setattr(cli.integration, 'integrate', lambda *_: {'rows': [], 'attempts': [], 'validation': {'status': 'FIXTURE_ONLY'}})
    metrics = {'computed_count': 0, 'reused_count': 0, 'resource_limit_global_indices': [], 'old_cache_writes': 0}
    def evaluate_missing(_integrated, _sources, target):
        target.mkdir()
        (target / 'fixture_cache.json').write_text('{"fixture":true}\n')
        return metrics
    monkeypatch.setattr(cli.integration, 'evaluate_missing', evaluate_missing)
    monkeypatch.setattr(cli.integration, 'cost_audit', lambda *_: {'historical_calls': 991, 'unknown_usage_holds': [], 'fixture': True})
    results = {'denominators': {'overall': {'planned_slots': 960}},
               'inference': {'seed': 0, 'bootstrap_replicates': 2000, 'signflip_replicates': 10000}, 'fixture': True}
    monkeypatch.setattr(cli.statistics, 'analyze_slots', lambda *_args, **_kwargs: results)
    monkeypatch.setattr(cli.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 1024**3))
    subprocess_calls = []
    def render_only(command, **kwargs):
        subprocess_calls.append(command)
        assert command[2:4] == ['-m', 'offline_evaluation.analysis.goal007.render']
        assert kwargs['check'] is True
        render_target = actual_path(command[command.index('--destination') + 1]); render_target.mkdir()
        (render_target / 'results.json').write_text(json.dumps(results))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.subprocess, 'run', render_only)
    args = SimpleNamespace(destination=str(destination), code_commit='a' * 40, audit_gate=str(gate_path),
                           rq2_audit=str(rq2_path), plot_python=str(tmp_path / 'fixture-plot-python'))
    # Do not mock cli.read_json: the inventory reference is intentionally a str,
    # and run() must convert it into Path at the actual reader boundary.
    report = cli.run(args)
    manifest = json.loads((destination / 'MANIFEST.json').read_text())
    assert report == manifest
    assert manifest['source_files'] == json.loads(inventory.read_text())['files']
    assert manifest['created_code_commit'] == args.code_commit
    assert manifest['source_inventories_before_after_equal'] is True
    assert manifest['all_slots_measured'] is False
    assert manifest['research_api_requests_sent'] == 0
    assert manifest['api_key_queries'] == 0
    assert manifest['github_mutations_by_analysis'] == 0
    assert manifest['missing_global_indices'] == [942]
    assert manifest['strict_metric_evaluation'] == metrics
    assert source.read_bytes() == original_source_bytes
    assert len(gate_calls) == 3 and len(subprocess_calls) == 1
    assert 'MANIFEST.json' not in manifest['products']
    for relative, descriptor in manifest['products'].items():
        path = destination / relative
        assert cli.sha256_file(path) == descriptor['sha256']
        assert path.stat().st_size == descriptor['bytes']
