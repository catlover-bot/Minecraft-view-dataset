"""CLI lifecycle, entirely isolated fixture state and fake remote storage."""
import json
import subprocess
from pathlib import Path

from offline_evaluation.experiments.goal015d import cli, runtime
from offline_evaluation.experiments.goal015d.tests.test_runtime import fixture_plan, MockAdapter, fixture_checkpoint
from offline_evaluation.experiments.goal015d.tests.test_checkpoint import FakeGithub


def commit():
    return subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()


def started(tmp_path, **kwargs):
    path = fixture_plan(tmp_path)
    fake = MockAdapter(**kwargs)
    result = runtime.execute(path,adapter=fake,confirmation=lambda p:True,
                             checkpoint_callback=fixture_checkpoint)
    return path,fake,result


def test_finish_retry_reuses_same_checkpoint_and_no_generation(tmp_path):
    path,fake,_ = started(tmp_path)
    github = FakeGithub(commit()); github.fail_download = True
    before = (path.parent/'enabled/ledger.json').read_bytes()
    failed = cli.finish(path,allow_create=True,github=github)
    assert failed['status'] == 'SAVE_FAILED' and failed['checkpoint']['path']
    github.fail_download = False
    result = cli.finish(path,github=github)
    assert result['status'] == 'COMPLETE_SAVED'
    assert result['checkpoint'] == failed['checkpoint']
    assert result['evaluation']['row_count'] == 16
    assert len(fake.sent) == 16 and (path.parent/'enabled/ledger.json').read_bytes() == before
    assert sum(stage=='release_create' for stage,_,_ in github.calls) == 1
    assert sum(stage=='asset_upload' for stage,_,_ in github.calls) == 1
    again = cli.finish(path,github=github)
    assert again['status'] == 'COMPLETE_SAVED'
    assert sum(stage=='asset_download' for stage,_,_ in github.calls) == 2


def test_save_only_reuses_failed_finish_and_no_evaluation(tmp_path,monkeypatch):
    path,fake,_ = started(tmp_path); github = FakeGithub(commit())
    github.fail_download = True
    first = cli.finish(path,allow_create=True,github=github)
    github.fail_download = False
    from offline_evaluation.experiments.goal015d import evaluate
    monkeypatch.setattr(evaluate,'evaluate_plan',lambda *a,**k: (_ for _ in ()).throw(AssertionError('no evaluation')))
    result = cli.save_only(path,github=github)
    assert result['status'] == 'SAVED_ONLY'
    assert result['checkpoint_dir'] == first['checkpoint']['path']
    assert len(fake.sent) == 16 and sum(stage=='asset_upload' for stage,_,_ in github.calls) == 1


def test_explicit_finish_partial_preserves_null_remaining(tmp_path):
    path,fake,state = started(tmp_path,transport_at=1)
    assert state['calls_uncertain'] == 1 and len(fake.sent) == 1
    result = cli.finish(path,allow_create=True,github=FakeGithub(commit()))
    assert result['status'] == 'PARTIAL_SAVED'
    assert result['calls_not_sent'] == 15 and result['calls_uncertain'] == 1
    assert result['evaluation']['row_count'] == 16 and len(fake.sent) == 1


def test_run_exactly_once_and_partial_nonzero(tmp_path,monkeypatch,capsys):
    path = fixture_plan(tmp_path); calls = []
    def execute(*args,**kwargs):
        calls.append(kwargs)
        return {'status':'SAFE_STOP','calls_settled':0,'calls_uncertain':1}
    monkeypatch.setattr(runtime,'execute',execute)
    monkeypatch.setattr(cli,'finish',lambda *a,**k: (_ for _ in ()).throw(AssertionError('no partial auto finish')))
    assert cli.main(['run','--plan',str(path),'--allow-create']) == 2
    assert len(calls) == 1 and calls[0]['resume'] is False
    assert json.loads(capsys.readouterr().out)['status'] == 'SAFE_STOP'


def test_run_save_failure_nonzero_and_explicit_upload_gate(tmp_path,monkeypatch,capsys):
    path = fixture_plan(tmp_path); calls = []
    monkeypatch.setattr(runtime,'execute',lambda *a,**k: {'status':'COMPLETE','calls_settled':16,'calls_uncertain':0})
    def finish(*args,**kwargs):
        calls.append(kwargs)
        return {'status':'SAVE_FAILED','generation_reexecuted':False}
    monkeypatch.setattr(cli,'finish',finish)
    assert cli.main(['run','--plan',str(path)]) == 2
    assert calls == [{'release_id':None,'allow_create':False}]
    assert json.loads(capsys.readouterr().out)['status'] == 'SAVE_FAILED'


def test_disabled_finish_does_not_create_authority_or_completion(tmp_path):
    path = fixture_plan(tmp_path)
    assert cli.main(['finish','--plan',str(path)]) == 2
    assert not (path.parent/'enabled').exists()
    assert not (path.parent/'completion.lock').exists()


def test_completion_lock_excludes_resume(tmp_path):
    path,fake,_ = started(tmp_path)
    import pytest
    with cli.completion_lock(runtime.verify_plan(path)):
        with pytest.raises(runtime.RuntimeErrorSafe,match='finish/save already active'):
            runtime.execute(path,resume=True,adapter=fake,checkpoint_callback=fixture_checkpoint)
    assert len(fake.sent) == 16
