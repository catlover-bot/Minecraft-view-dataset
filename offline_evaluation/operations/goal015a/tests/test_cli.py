"""CLI sequencing fixtures; no provider/credential/network access."""
import json

import pytest

from offline_evaluation.operations.goal015a import cli, runtime


def test_completion_save_lock_refuses_parallel_finish(tmp_path):
    with cli._completion_lock(tmp_path):
        with pytest.raises(runtime.RuntimeErrorSafe, match='already active'):
            with cli._completion_lock(tmp_path):
                pytest.fail('parallel completion entered')
    with cli._completion_lock(tmp_path):
        pass


def test_run_partial_exits_nonzero_without_evaluate_or_upload(monkeypatch, capsys):
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: {
        'status': 'SAFE_STOP', 'calls_settled': 7, 'calls_not_sent': 0, 'calls_uncertain': 1})
    monkeypatch.setattr(cli, 'finish', lambda *a, **k: pytest.fail('partial must not report completed saved workflow'))
    assert cli.main(['run', '--plan', 'fixture', '--pricing-evidence', 'fixture-evidence']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'SAFE_STOP'


def test_run_complete_runs_finish_once_and_never_requests_resume(monkeypatch, capsys):
    actions = []
    def execute(path, **kwargs):
        actions.append(('execute', kwargs))
        return {'status': 'COMPLETE', 'calls_settled': 8, 'calls_not_sent': 0, 'calls_uncertain': 0}
    def finish(path, **kwargs):
        actions.append(('finish', kwargs))
        return {'status': 'COMPLETE_SAVED', 'settled_attempts': 80, 'reconstruction_success_not_implied': True}
    monkeypatch.setattr(runtime, 'execute', execute)
    monkeypatch.setattr(cli, 'finish', finish)
    assert cli.main(['run', '--plan', 'fixture', '--pricing-evidence', 'fixture-evidence']) == 0
    assert [x[0] for x in actions] == ['execute', 'finish']
    assert actions[0][1]['resume'] is False and actions[1][1]['allow_create'] is True
    assert json.loads(capsys.readouterr().out)['reconstruction_success_not_implied']


def test_save_failure_propagates_and_does_not_restart(monkeypatch):
    sent = []
    def execute(*args, **kwargs):
        sent.append(1)
        return {'status': 'COMPLETE', 'calls_settled': 8, 'calls_not_sent': 0, 'calls_uncertain': 0}
    def finish(*args, **kwargs):
        raise RuntimeError('fixture save interrupted')
    monkeypatch.setattr(runtime, 'execute', execute)
    monkeypatch.setattr(cli, 'finish', finish)
    with pytest.raises(RuntimeError, match='fixture save interrupted'):
        cli.main(['run', '--plan', 'fixture', '--pricing-evidence', 'fixture-evidence'])
    assert len(sent) == 1


def test_finish_reuses_checkpoint_after_failed_upload_without_generation(tmp_path, monkeypatch):
    from offline_evaluation.operations.goal015a import evaluate, checkpoint
    from offline_evaluation.experiments.goal015 import runtime as parent
    root = tmp_path / 'child'; (root/'enabled').mkdir(parents=True)
    parent._save(root/'enabled/ledger.json', {'fixture': 'stable-ledger'})
    evaldir = tmp_path/'evaluation'; evaldir.mkdir()
    parent._save(evaldir/'evaluation_80.json', {'rows': 'fixture'})
    summary = {'evaluation_artifact': parent.descriptor(evaldir/'evaluation_80.json'),
               'evaluated_protocol_valid_rows': 48, 'actual_metric_rows': 48}
    parent._save(evaldir/'summary.json', summary)
    monkeypatch.setattr(runtime, 'verify_plan', lambda p: {'plan_id': 'fixture-plan', 'canonical_root': str(root)})
    monkeypatch.setattr(runtime, 'status', lambda p: {'status': 'COMPLETE', 'calls_settled': 8,
        'calls_not_sent': 0, 'calls_uncertain': 0})
    monkeypatch.setattr(runtime, 'execute', lambda *a, **k: pytest.fail('finish must never generate'))
    monkeypatch.setattr(evaluate, 'evaluate_plan', lambda p: summary)
    monkeypatch.setattr(evaluate, 'find_existing_evaluation', lambda p: summary)
    checkpoints, uploads = [], []
    def save(p):
        checkpoints.append(1)
        directory = root/'checkpoint'; directory.mkdir()
        parent._save(directory/'checkpoint.json', {'fixture': 'immutable-checkpoint'})
        return {'path': str(directory)}
    def upload(directory, verification, **kwargs):
        uploads.append((directory, str(verification)))
        if len(uploads) == 1:
            raise RuntimeError('fixture download interrupted')
        return {'status': 'GITHUB_VERIFIED', 'release_id': 123, 'asset_id': 456}
    monkeypatch.setattr(checkpoint, 'create_save_only', save)
    monkeypatch.setattr(checkpoint, 'upload_checkpoint', upload)
    with pytest.raises(RuntimeError):
        cli.finish(root/'plan.json', allow_create=True)
    done = cli.finish(root/'plan.json', allow_create=True)
    assert done['status'] == 'COMPLETE_SAVED' and done['reconstruction_success_not_implied']
    assert len(checkpoints) == 1 and uploads[0][0] == uploads[1][0]
    assert uploads[0][1] != uploads[1][1]


def test_actual_cli_chain_eight_fake_sends_then_save_only_retry(tmp_path, monkeypatch, capsys):
    from offline_evaluation.operations.goal015a import checkpoint
    from offline_evaluation.operations.goal015a.tests.test_evaluate_checkpoint import fixture_plan, FakeGithub
    from offline_evaluation.operations.goal015a.tests.test_runtime import MockAdapter
    path = fixture_plan(tmp_path, monkeypatch)
    adapter = MockAdapter()
    actual_execute, actual_upload = runtime.execute, checkpoint.upload_checkpoint
    executions, packages, github = [], [], []
    def execute(plan_path, **kwargs):
        executions.append(kwargs)
        return actual_execute(plan_path, adapter=adapter, confirmation=lambda _: True, **kwargs)
    def upload(directory, verification, **kwargs):
        package = runtime.read_json(directory + '/checkpoint.json')
        packages.append(package['checkpoint_id'])
        if not github:
            github.append(FakeGithub(package['target_commitish']))
            github[0].fail_download = True
        return actual_upload(directory, verification, github=github[0], **kwargs)
    monkeypatch.setattr(runtime, 'execute', execute)
    monkeypatch.setattr(checkpoint, 'upload_checkpoint', upload)
    with pytest.raises(RuntimeError, match='fixture interruption'):
        cli.main(['run', '--plan', str(path), '--pricing-evidence',
                  runtime.read_json(path)['pricing_evidence']['path']])
    assert adapter.calls == 8 and len(executions) == 1
    github[0].fail_download = False
    assert cli.main(['finish', '--plan', str(path), '--allow-create']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'COMPLETE_SAVED' and result['settled_attempts'] == 80
    assert result['reconstruction_success_not_implied'] and result['actual_metric_rows'] == 79
    assert adapter.calls == 8 and len(executions) == 1 and packages[0] == packages[1]
    assert sum(stage == 'release_create' for stage, _, _ in github[0].calls) == 1
    assert sum(stage == 'asset_upload' for stage, _, _ in github[0].calls) == 1
    runtime._save(tmp_path/'MOCK_CLI_COMPLETION_RECEIPT.json', {
        'status': 'PASS', 'fixture_only': True, 'research_api_requests_sent': 0,
        'fake_generation_calls': adapter.calls, 'generation_invocations': len(executions),
        'fake_drafts_created': 1, 'fake_assets_uploaded': 1,
        'checkpoint_reused_after_failed_download': packages[0] == packages[1],
        'completion_status': result['status'], 'settled_attempts': 80,
    }, exclusive=True)
