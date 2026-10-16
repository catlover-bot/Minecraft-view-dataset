"""CPU-only CLI gates and genuine controlling-terminal approval for synthetic plans."""
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import pty
import select
import signal
import time

import pytest

from offline_evaluation.operations.goal006c import cli, runner
from offline_evaluation.server_v2.generation import GenerationError


def snapshot(f):
    return {str(p):('dir' if p.is_dir() else p.read_bytes()) for p in f['root'].parent.rglob('*')}


def forbid(monkeypatch):
    def fail(*args,**kwargs):pytest.fail('preview/refusal must never activate or run')
    monkeypatch.setattr(runner,'initialize_enabled',fail);monkeypatch.setattr(runner,'run',fail)


def test_preview_and_disabled_checkpoint_do_not_create_state(fixture_plan,monkeypatch,capsys):
    forbid(monkeypatch);before=snapshot(fixture_plan)
    assert cli.main(['--plan',str(fixture_plan['path']),'--preview'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['approved'] is False and result['supplemental_not_run']==17
    assert result['historical_unknown_usage_attempts_preserved']==2
    assert result['combined_calls']==974 and result['pricing_current_utc_day'] is True
    assert result['required_new_approval_scope']==runner.APPROVAL_SCOPE
    assert cli.main(['--plan',str(fixture_plan['path']),'--checkpoint'])==5
    assert 'not_enabled' in capsys.readouterr().err
    assert snapshot(fixture_plan)==before


def test_no_tty_execute_uses_no_stdin_fallback(fixture_plan,monkeypatch,capsys):
    forbid(monkeypatch);before=snapshot(fixture_plan)
    def no_tty(*a,**k):raise OSError(errno.ENXIO,'PRIVATE_FIXTURE_ERROR')
    monkeypatch.setattr(cli,'open',no_tty,raising=False)
    assert cli.main(['--plan',str(fixture_plan['path']),'--execute'])==5
    output=capsys.readouterr();assert json.loads(output.err)['code']=='goal006c_tty_unavailable_no_activation'
    assert 'PRIVATE_FIXTURE_ERROR' not in output.err and snapshot(fixture_plan)==before


def test_missing_keys_does_not_open_tty(fixture_plan,monkeypatch,capsys):
    forbid(monkeypatch);monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    monkeypatch.setattr(cli,'tty_approval',lambda *a:pytest.fail('no TTY before keys gate'))
    assert cli.main(['--plan',str(fixture_plan['path']),'--execute'])==5
    assert 'both_provider_keys_required' in capsys.readouterr().err


def read_until(fd, marker, deadline):
    data=b''
    while marker not in data:
        left=deadline-time.monotonic();assert left>0,'PTY deadline'
        assert select.select([fd],[],[],left)[0],'PTY timeout'
        chunk=os.read(fd,65536);assert chunk,'PTY child stopped'
        data+=chunk
    return data


@contextmanager
def tty_child(plan):
    read_fd,write_fd=os.pipe();pid,terminal=pty.fork()
    if pid==0:
        try:
            os.close(read_fd)
            signal.signal(signal.SIGINT,signal.default_int_handler)
            try:result={'approval':cli.tty_approval(plan)}
            except GenerationError as exc:result={'error':str(exc)}
            os.write(write_fd,json.dumps(result).encode()+b'\n');os.close(write_fd);os._exit(0)
        except BaseException:os._exit(127)
    os.close(write_fd)
    try:yield pid,terminal,read_fd
    finally:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:os.waitpid(pid,0)
        except ChildProcessError:pass
        os.close(terminal);os.close(read_fd)


@pytest.mark.parametrize('answer_kind',['exact','space','mismatch','eof','partial_eof','interrupt'])
def test_actual_tty_new_scope_exact_line_only(fixture_plan,answer_kind):
    plan=fixture_plan['plan'];exact=cli.confirmation_text(plan).encode();before=snapshot(fixture_plan)
    with tty_child(plan) as (pid,terminal,fd):
        prompt=read_until(terminal,b'> ',time.monotonic()+5)
        assert exact in prompt and b'17' in prompt and b'NO 520 RETRY' in prompt
        assert not select.select([fd],[],[],0.05)[0]
        answers={'exact':exact+b'\n','space':exact+b' \n','mismatch':b'APPROVE GOAL006B\n',
                 'eof':b'\x04','partial_eof':exact+b'\x04\x04','interrupt':b'\x03'}
        os.write(terminal,answers[answer_kind]);result=json.loads(read_until(fd,b'\n',time.monotonic()+5))
    if answer_kind=='exact':
        approval=result['approval'];runner._approval(plan,approval,fresh=True)
        assert approval['acknowledges_both_unknown_holds'] is True
        assert approval['scope']==runner.APPROVAL_SCOPE
    else:assert 'approval' not in result and result['error'].endswith('no_activation')
    assert snapshot(fixture_plan)==before


def test_save_only_never_enters_generation_or_tty(monkeypatch,tmp_path,capsys):
    forbid(monkeypatch)
    monkeypatch.setattr(cli,'tty_approval',lambda *a:pytest.fail('save-only must not approve'))
    calls=[]
    def save(directory,verification_directory,**kwargs):
        calls.append((directory,verification_directory,kwargs));return {'status':'GITHUB_VERIFIED','research_api_requests_sent':0}
    monkeypatch.setattr(cli,'save_checkpoint_only',save)
    assert cli.main(['--save-checkpoint',str(tmp_path/'package'),'--verification-directory',str(tmp_path/'verify'),'--allow-create'])==0
    assert calls[0][2]=={'release_id':None,'allow_create':True}
    assert json.loads(capsys.readouterr().out)['status']=='GITHUB_VERIFIED'
