"""Shell lifecycle checks on rewritten copies, never the canonical launcher.

Each copy uses an isolated root, fake Python CLI/preflight, fake tmux lookup,
synthetic environment and fake credentials. The embedded production preflight
is parsed statically; its real guards have separate runtime contract tests.
"""
import ast
import json
import os
from pathlib import Path
import pty
import select
import shlex
import subprocess
import sys
import time

import pytest


SOURCE = Path(__file__).parents[1] / 'start-reviewed.sh'
FAKE_KEY_OPENAI = 'fixture-only-openai-value'
FAKE_KEY_ANTHROPIC = 'fixture-only-anthropic-value'


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o700)
    return path


def _fixture(tmp_path, *, existing=(), cli_status=0, preflight_failure=0,
             tmux_mismatch=False, tee_failure=False):
    root = tmp_path / 'run'; root.mkdir(); (root / 'tmp').mkdir()
    repo = tmp_path / 'repo'; repo.mkdir()
    evidence = root / 'reviewed.json'; evidence.write_text('{}')
    events = root / 'events.jsonl'
    fake_python = _executable(tmp_path / 'fake-python', '#!' + sys.executable + '\n' + r'''
import json, os, sys
from pathlib import Path
events = Path(os.environ['FIXTURE_EVENTS'])
old = events.read_text().splitlines() if events.exists() else []
preflight = sys.argv[1:3] == ['-B', '-']
event = {'stage': 'preflight' if preflight else 'cli',
         'openai_present': bool(os.environ.get('OPENAI_API_KEY')),
         'anthropic_present': bool(os.environ.get('ANTHROPIC_API_KEY'))}
with events.open('a') as target:
    target.write(json.dumps(event) + '\n')
if preflight:
    sys.stdin.read()  # Never execute the original inline Python preflight.
    count = 1 + sum(json.loads(line)['stage'] == 'preflight' for line in old)
    if count == int(os.environ['FIXTURE_PREFLIGHT_FAILURE']):
        print('Synthetic preflight refused.')
        raise SystemExit(1)
    print('Synthetic preflight passed.')
    raise SystemExit(0)
assert sys.argv[1:5] == ['-B', '-m', 'offline_evaluation.operations.goal015a.cli', 'run']
assert sys.argv[5:] == ['--plan', os.environ['FIXTURE_ROOT'] + '/plan.json',
                       '--pricing-evidence', os.environ['FIXTURE_ROOT'] + '/reviewed.json']
assert os.environ['OPENAI_API_KEY'] == 'fixture-only-openai-value'
assert os.environ['ANTHROPIC_API_KEY'] == 'fixture-only-anthropic-value'
print('FAKE_CLI_EXECUTED_ONCE')
status = int(os.environ['FIXTURE_CLI_STATUS'])
print(json.dumps({'status': {0: 'COMPLETE_SAVED', 1: 'SAVE_FAILED', 2: 'SAFE_STOP'}[status],
                  'fixture_only': True}))
raise SystemExit(status)
''')
    fake_tmux = _executable(tmp_path / 'fake-tmux', '#!/bin/bash\n' + (
        "printf '%s\\n' /dev/fixture-wrong-pane\n" if tmux_mismatch else
        'tty=$(/bin/ps -o tty= -p "$$")\ntty="${tty//[[:space:]]/}"\nprintf "/dev/%s\\n" "$tty"\n'))
    source = SOURCE.read_text()
    replacements = {
        "readonly goal015a_repo='/root/workspace/Minecraft-view-dataset'": 'readonly goal015a_repo=' + shlex.quote(str(repo)),
        "readonly goal015a_python='/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python'": 'readonly goal015a_python=' + shlex.quote(str(fake_python)),
        "readonly goal015a_root='/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79'": 'readonly goal015a_root=' + shlex.quote(str(root)),
        '/usr/bin/tmux': shlex.quote(str(fake_tmux)),
    }
    evidence_line = next(line for line in source.splitlines() if line.startswith('readonly goal015a_evidence='))
    replacements[evidence_line] = 'readonly goal015a_evidence=' + shlex.quote(str(evidence))
    for old, new in replacements.items():
        assert old in source
        source = source.replace(old, new)
    if tee_failure:
        fake_tee = _executable(tmp_path / 'fake-tee', '#!' + sys.executable + '\n' + r'''
import sys
from pathlib import Path
data = sys.stdin.buffer.read()
assert sys.argv[1] == '-a'
with Path(sys.argv[2]).open('ab') as target:
    target.write(data)
sys.stdout.buffer.write(data)
raise SystemExit(9 if b'FAKE_CLI_EXECUTED_ONCE' in data else 0)
''')
        source = source.replace('/usr/bin/tee', shlex.quote(str(fake_tee)))
    assert '/root/minecraft-runs/goal015-' not in source
    assert "readonly goal015a_root='/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79'" not in source
    assert '/root/workspace/' not in source
    script = tmp_path / 'launcher-fixture.sh'; script.write_text(source)
    # Deliberately do not inherit or inspect the caller's environment or keys.
    env = {'PATH': '/usr/bin:/bin', 'TERM': 'xterm', 'TMUX': 'fixture-only', 'TMUX_PANE': '%1',
           'PYTHONDONTWRITEBYTECODE': '1', 'FIXTURE_EVENTS': str(events), 'FIXTURE_ROOT': str(root),
           'FIXTURE_CLI_STATUS': str(cli_status), 'FIXTURE_PREFLIGHT_FAILURE': str(preflight_failure)}
    for name in existing:
        env[name] = {'OPENAI_API_KEY': FAKE_KEY_OPENAI, 'ANTHROPIC_API_KEY': FAKE_KEY_ANTHROPIC}[name]
    return script, root, events, env


def _pty_run(script, env, *, empty_key=False):
    pid, master = pty.fork()
    if pid == 0:
        os.execve('/bin/bash', ['bash', str(script)], env)
    output = b''; sent = set(); deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], .02)
            if ready:
                try:
                    output += os.read(master, 65536)
                except OSError:
                    pass
            for name, value in [('OPENAI_API_KEY', FAKE_KEY_OPENAI), ('ANTHROPIC_API_KEY', FAKE_KEY_ANTHROPIC)]:
                if (name + ' (hidden; empty/EOF cancels): ').encode() in output and name not in sent:
                    time.sleep(.02)
                    os.write(master, ('\n' if empty_key else value + '\n').encode())
                    sent.add(name)
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                # Drain final trap output after the process exits.
                while select.select([master], [], [], 0)[0]:
                    try:
                        part = os.read(master, 65536)
                    except OSError:
                        break
                    if not part:
                        break
                    output += part
                return os.waitstatus_to_exitcode(status), output.decode(), sent
        os.kill(pid, 9); os.waitpid(pid, 0)
        raise AssertionError('Isolated launcher PTY timed out')
    finally:
        os.close(master)


def _evidence(root, events, output):
    logs = list(root.glob('start-reviewed-*.log'))
    assert len(logs) == 1 and logs[0].stat().st_mode & 0o777 == 0o600
    log = logs[0].read_text()
    assert str(logs[0]) in output and str(logs[0]) in log
    assert FAKE_KEY_OPENAI not in output + log
    assert FAKE_KEY_ANTHROPIC not in output + log
    assert not (root / 'enabled').exists() and not (root / 'execution.lock').exists()
    return [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []


def test_production_launcher_static_only():
    result = subprocess.run(['/bin/bash', '-n', str(SOURCE)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    source = SOURCE.read_text()
    inline = source.split("<<'PY'", 1)[1].split('\n', 1)[1].split('\nPY\n', 1)[0]
    ast.parse(inline)
    assert source.count('offline_evaluation.operations.goal015a.cli run') == 1
    assert source.index('goal015a_preflight ||') < source.index('goal015a_require_key OPENAI_API_KEY')
    assert 'goal015a_pipeline_status=("${PIPESTATUS[@]}")' in source


@pytest.mark.parametrize('existing', [(), ('OPENAI_API_KEY',), ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY')])
def test_real_pty_only_missing_keys_hidden_then_single_run(tmp_path, existing):
    script, root, events, env = _fixture(tmp_path, existing=existing)
    code, output, prompted = _pty_run(script, env)
    rows = _evidence(root, events, output)
    assert code == 0 and prompted == {'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'} - set(existing)
    assert [row['stage'] for row in rows] == ['preflight', 'preflight', 'cli']
    assert rows[0]['openai_present'] == ('OPENAI_API_KEY' in existing)
    assert rows[0]['anthropic_present'] == ('ANTHROPIC_API_KEY' in existing)
    assert rows[1]['openai_present'] and rows[1]['anthropic_present']
    assert output.count('FAKE_CLI_EXECUTED_ONCE') == 1 and 'Wrapper exit status: 0' in output


@pytest.mark.parametrize('failure', [1, 2])
def test_preflight_failure_prevents_cli(tmp_path, failure):
    script, root, events, env = _fixture(tmp_path, preflight_failure=failure)
    code, output, prompted = _pty_run(script, env)
    rows = _evidence(root, events, output)
    assert code != 0 and [row['stage'] for row in rows] == ['preflight'] * failure
    assert bool(prompted) == (failure == 2)
    assert 'FAKE_CLI_EXECUTED_ONCE' not in output


def test_actual_controlling_tty_must_match_tmux(tmp_path):
    script, root, events, env = _fixture(tmp_path, tmux_mismatch=True)
    code, output, prompted = _pty_run(script, env)
    assert code != 0 and not prompted and not _evidence(root, events, output)
    assert 'controlling TTY differ' in output


@pytest.mark.parametrize('status', [1, 2])
def test_partial_or_save_failure_propagates_without_retry(tmp_path, status):
    script, root, events, env = _fixture(tmp_path, cli_status=status)
    code, output, _ = _pty_run(script, env)
    rows = _evidence(root, events, output)
    assert code == status and sum(row['stage'] == 'cli' for row in rows) == 1
    assert f'Wrapper exit status: {status}' in output
    assert '"status": "COMPLETE_SAVED"' not in output


def test_log_writer_failure_is_nonzero_after_single_cli(tmp_path):
    script, root, events, env = _fixture(tmp_path, tee_failure=True)
    code, output, _ = _pty_run(script, env)
    rows = _evidence(root, events, output)
    assert code == 74 and sum(row['stage'] == 'cli' for row in rows) == 1
    assert 'CLI process exit status: 0; log writer exit status: 9' in output
    assert 'Wrapper exit status: 74' in output


def test_empty_hidden_entry_cancels_before_second_preflight(tmp_path):
    script, root, events, env = _fixture(tmp_path)
    code, output, _ = _pty_run(script, env, empty_key=True)
    rows = _evidence(root, events, output)
    assert code != 0 and [row['stage'] for row in rows] == ['preflight']
    assert 'Empty credential entry; no launch.' in output


def test_no_controlling_tty_refuses_before_preflight(tmp_path):
    script, root, events, env = _fixture(tmp_path)
    result = subprocess.run(['/bin/bash', str(script)], env=env, capture_output=True, start_new_session=True)
    output = (result.stdout + result.stderr).decode()
    assert result.returncode != 0 and not _evidence(root, events, output)
    assert 'A real controlling terminal is required.' in output
