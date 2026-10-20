"""Isolated launcher clones with a real PTY; no production execution or keys."""
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

SOURCE = Path(__file__).parents[1] / "start-reviewed.sh"
KEYS = {"OPENAI_API_KEY": "isolated-fixture-openai",
        "ANTHROPIC_API_KEY": "isolated-fixture-anthropic"}


def fixture_launcher(tmp_path, *, failure=0, result=0, existing=(), renewal=False,
                     require_approval=False, tee_failure=False):
    root = tmp_path / "run"
    root.mkdir()
    (root / "tmp").mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    events = root / "events.jsonl"
    fake = tmp_path / "fake-python"
    fake.write_text("#!" + sys.executable + "\n" + r'''
import json, os, sys
from pathlib import Path
assert sys.argv[1:4] == ['-B', '-m', 'offline_evaluation.experiments.goal015d.cli']
stage = sys.argv[4]
events = Path(os.environ['FIXTURE_EVENTS'])
previous = [json.loads(x) for x in events.read_text().splitlines()] if events.exists() else []
assert stage in ('preflight', 'run')
assert sys.argv[5:7] == ['--plan', os.environ['FIXTURE_ROOT'] + '/plan.json']
rest = sys.argv[7:]
if os.environ['FIXTURE_RENEWAL']:
    assert rest[:2] == ['--pricing-evidence', os.environ['FIXTURE_RENEWAL']]
    rest = rest[2:]
assert rest == (['--require-unstarted'] if stage == 'preflight' else ['--allow-create'])
with events.open('a') as stream:
    stream.write(json.dumps({'stage': stage, 'args': sys.argv[5:],
        'keys_present': {key: bool(os.environ.get(key)) for key in ('OPENAI_API_KEY','ANTHROPIC_API_KEY')}}) + '\n')
if stage == 'preflight':
    count = 1 + sum(x['stage'] == stage for x in previous)
    if count == int(os.environ['FIXTURE_PREFLIGHT_FAILURE']):
        print('Synthetic preflight rejected.')
        raise SystemExit(1)
    print('Synthetic preflight passed.')
    raise SystemExit(0)
assert os.environ['OPENAI_API_KEY'] == 'isolated-fixture-openai'
assert os.environ['ANTHROPIC_API_KEY'] == 'isolated-fixture-anthropic'
if os.environ['FIXTURE_REQUIRE_APPROVAL'] == '1':
    with open('/dev/tty', 'r') as reader, open('/dev/tty', 'w') as writer:
        assert reader.isatty() and writer.isatty()
        writer.write('Fixture exact approval required: ')
        writer.flush()
        if reader.readline().rstrip('\n') != 'APPROVE FIXTURE ONLY':
            print('Synthetic approval cancelled.')
            raise SystemExit(3)
print('FAKE_WORKFLOW_ONCE')
status = int(os.environ['FIXTURE_RESULT'])
print(json.dumps({'status': {0:'COMPLETE_SAVED',1:'SAVE_FAILED',2:'SAFE_STOP'}[status],
                  'research_api_requests_sent':0,'fixture_only':True}))
raise SystemExit(status)
''')
    fake.chmod(0o700)
    text = SOURCE.read_text()
    for marker, value in {"__GOAL015D_REPO__": repo,
                          "__GOAL015D_PYTHON__": fake, "__GOAL015D_ROOT__": root}.items():
        text = text.replace("'" + marker + "'", shlex.quote(str(value)))
    if tee_failure:
        tee = tmp_path / "fake-tee"
        tee.write_text("#!" + sys.executable + "\n" + r'''
import sys
from pathlib import Path
data=sys.stdin.buffer.read()
with Path(sys.argv[2]).open('ab') as f:f.write(data)
sys.stdout.buffer.write(data)
raise SystemExit(9 if b'FAKE_WORKFLOW_ONCE' in data else 0)
''')
        tee.chmod(0o700)
        text = text.replace("/usr/bin/tee", shlex.quote(str(tee)))
    script = root / "start-reviewed.sh"
    script.write_text(text)
    # Do not inherit or inspect the calling process's environment/credentials.
    env = {"PATH": "/usr/bin:/bin", "TERM": "xterm", "PYTHONDONTWRITEBYTECODE": "1",
           "FIXTURE_EVENTS": str(events), "FIXTURE_ROOT": str(root),
           "FIXTURE_PREFLIGHT_FAILURE": str(failure), "FIXTURE_RESULT": str(result),
           "FIXTURE_REQUIRE_APPROVAL": str(int(require_approval)), "FIXTURE_RENEWAL": ""}
    for key in existing:
        env[key] = KEYS[key]
    args = []
    if renewal:
        evidence = root / "same contract reviewed.json"
        evidence.write_text("{}")
        env["FIXTURE_RENEWAL"] = str(evidence)
        args = ["--pricing-evidence", str(evidence)]
    return script, root, events, env, args


def run_pty(script, env, args=(), *, credential="normal", approval="APPROVE FIXTURE ONLY\n"):
    pid, master = pty.fork()
    if pid == 0:
        os.execve("/bin/bash", ["bash", str(script), *args], env)
    output = b""
    responded = set()
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], .02)[0]:
                try:
                    output += os.read(master, 65536)
                except OSError:
                    pass
            for key, value in KEYS.items():
                if (key + " (hidden; empty/EOF cancels): ").encode() in output and key not in responded:
                    time.sleep(.02)
                    text = value + "\n" if credential == "normal" else "\n" if credential == "empty" else "\x04"
                    os.write(master, text.encode())
                    responded.add(key)
            if b"Fixture exact approval required: " in output and "approval" not in responded:
                os.write(master, approval.encode())
                responded.add("approval")
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                while select.select([master], [], [], 0)[0]:
                    try:
                        part = os.read(master, 65536)
                    except OSError:
                        break
                    if not part:
                        break
                    output += part
                return os.waitstatus_to_exitcode(status), output.decode(), responded
        os.kill(pid, 9)
        os.waitpid(pid, 0)
        raise AssertionError("isolated launcher timeout")
    finally:
        os.close(master)


def evidence(root, events, output):
    logs = list(root.glob("start-reviewed-*.log"))
    assert len(logs) == 1 and logs[0].stat().st_mode & 0o777 == 0o600
    log = logs[0].read_text()
    assert str(logs[0]) in output and str(logs[0]) in log
    for value in KEYS.values():
        assert value not in output + log
    assert not (root / "enabled").exists()
    return [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []


def test_template_syntax_only_and_explicit_gates():
    result = subprocess.run(["/bin/bash", "-n", str(SOURCE)], capture_output=True)
    assert result.returncode == 0
    text = SOURCE.read_text()
    assert text.index("goal015d_preflight ||") < text.index("goal015d_require_key OPENAI_API_KEY")
    assert text.count("offline_evaluation.experiments.goal015d.cli run") == 1
    assert "--allow-create" in text and "PIPESTATUS" in text
    assert "goal015d_tty_read}</dev/tty" in text and "goal015d_tty_write}>/dev/tty" in text


@pytest.mark.parametrize("existing", [(), ("OPENAI_API_KEY",), tuple(KEYS)])
def test_hidden_missing_keys_two_preflights_and_one_run(tmp_path, existing):
    script, root, events, env, args = fixture_launcher(tmp_path, existing=existing, renewal=True)
    code, output, prompted = run_pty(script, env, args)
    rows = evidence(root, events, output)
    assert code == 0 and prompted == set(KEYS) - set(existing)
    assert [row["stage"] for row in rows] == ["preflight", "preflight", "run"]
    assert output.count("FAKE_WORKFLOW_ONCE") == 1
    assert all(rows[1]["keys_present"].values())


@pytest.mark.parametrize("failure", [1, 2])
def test_preflight_rejection_prevents_run(tmp_path, failure):
    script, root, events, env, args = fixture_launcher(tmp_path, failure=failure)
    code, output, prompted = run_pty(script, env, args)
    rows = evidence(root, events, output)
    assert code != 0 and [row["stage"] for row in rows] == ["preflight"] * failure
    assert bool(prompted) == (failure == 2)
    assert "FAKE_WORKFLOW_ONCE" not in output


@pytest.mark.parametrize("credential", ["empty", "eof"])
def test_empty_or_eof_key_cancels(tmp_path, credential):
    script, root, events, env, args = fixture_launcher(tmp_path)
    code, output, _ = run_pty(script, env, args, credential=credential)
    assert code != 0 and [row["stage"] for row in evidence(root, events, output)] == ["preflight"]


@pytest.mark.parametrize("approval", ["\n", "\x04", "wrong\n"])
def test_real_pty_approval_not_automatically_entered(tmp_path, approval):
    script, root, events, env, args = fixture_launcher(tmp_path, require_approval=True)
    code, output, prompted = run_pty(script, env, args, approval=approval)
    rows = evidence(root, events, output)
    assert code == 3 and "approval" in prompted and sum(row["stage"] == "run" for row in rows) == 1
    assert "FAKE_WORKFLOW_ONCE" not in output


@pytest.mark.parametrize("status", [1, 2])
def test_save_failure_or_partial_propagates_without_retry(tmp_path, status):
    script, root, events, env, args = fixture_launcher(tmp_path, result=status)
    code, output, _ = run_pty(script, env, args)
    rows = evidence(root, events, output)
    assert code == status and sum(row["stage"] == "run" for row in rows) == 1
    assert '"status": "COMPLETE_SAVED"' not in output


def test_tee_failure_overrides_success(tmp_path):
    script, root, events, env, args = fixture_launcher(tmp_path, tee_failure=True)
    code, output, _ = run_pty(script, env, args)
    assert code == 74 and sum(row["stage"] == "run" for row in evidence(root, events, output)) == 1


def test_no_controlling_tty_refuses(tmp_path):
    script, root, events, env, args = fixture_launcher(tmp_path)
    result = subprocess.run(["/bin/bash", str(script), *args], env=env,
                            capture_output=True, start_new_session=True)
    output = (result.stdout + result.stderr).decode()
    assert result.returncode != 0 and not evidence(root, events, output)


def test_copied_launcher_refuses_before_prompts(tmp_path):
    script, root, events, env, args = fixture_launcher(tmp_path)
    copied = tmp_path / "copied.sh"
    copied.write_bytes(script.read_bytes())
    code, output, prompted = run_pty(copied, env, args)
    assert code == 73 and not prompted and not events.exists()
    assert not list(root.glob("start-reviewed-*.log"))
