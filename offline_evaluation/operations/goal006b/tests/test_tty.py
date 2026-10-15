"""Real controlling PTY coverage using disabled synthetic plans and blocked HTTP.

Fork children inherit the explicit fixture-only source validator and network
guards from conftest. The actual CLI, TTY approval, activation, runner, claims,
ledger and native response parser run. Only provider/checkpoint transports are
injected; no production plan or credential is used.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import errno
import io
import json
import os
from pathlib import Path
import pty
import select
import signal
import sys
import time

import pytest

from offline_evaluation.batch_v1.tests.test_runner import transports
from offline_evaluation.operations.goal006b import cli, runner
from offline_evaluation.operations.goal006b.tests.test_cli import tree_snapshot
from offline_evaluation.operations.goal006b.tests.test_runner import verified_checkpoint, saved
from offline_evaluation.server_v2.generation import read_json

TIMEOUT = 12.0
PRIVATE = "PRIVATE_FIXTURE_TERMINAL_ERROR_NEVER_REPORT"


def assert_originals(fixture):
    assert all(Path(path).read_bytes() == data for path, data in fixture["source_before"].items())


def child_main(fixture, result_fd, *, stdin_mode="tty", fault="none"):
    calls, opened = [], []
    stdout, stderr = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = stdout, stderr
    # Keep a disconnected PTY observable as an I/O failure, rather than allowing
    # SIGHUP's default process termination to obscure the fail-closed result.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    exact = cli.confirmation_text(fixture["plan"])
    if stdin_mode == "null":
        fd = os.open("/dev/null", os.O_RDONLY)
        os.dup2(fd, 0); os.close(fd)
    elif stdin_mode in {"pipe", "environment"}:
        read_fd, write_fd = os.pipe()
        os.write(write_fd, (exact + "\n").encode())
        os.close(write_fd)
        os.dup2(read_fd, 0); os.close(read_fd)
    if stdin_mode == "environment":
        for name in ("GOAL006B_APPROVAL", "GOAL006B_CONFIRMATION", "APPROVAL", "CONFIRMATION"):
            os.environ[name] = exact

    real_run = runner.run
    def fixture_run(path, **kwargs):
        return real_run(path, **kwargs, transports=transports(calls),
                        checkpoint_verifier=verified_checkpoint, checkpoint_callback=saved)
    runner.run = fixture_run

    class Terminal:
        def __init__(self, terminal, mode):
            self.terminal, self.mode = terminal, mode
        def __enter__(self):
            self.terminal.__enter__()
            return self
        def __exit__(self, *args):
            result = self.terminal.__exit__(*args)
            if fault == "close_" + self.mode:
                raise OSError(errno.EIO, PRIVATE)
            return result
        def isatty(self):
            return False if fault == "not_tty_" + self.mode else self.terminal.isatty()
        def write(self, value):
            if fault == "write":
                raise OSError(errno.EIO, PRIVATE)
            return self.terminal.write(value)
        def flush(self):
            if fault == "flush":
                raise OSError(errno.EIO, PRIVATE)
            return self.terminal.flush()
        def readline(self):
            if fault == "read":
                raise OSError(errno.EIO, PRIVATE)
            if fault == "read_eof":
                raise EOFError(PRIVATE)
            return self.terminal.readline()

    def terminal_open(path, mode="r", *args, **kwargs):
        assert path == "/dev/tty"
        opened.append({"mode": mode, "encoding": kwargs.get("encoding")})
        if fault == "open_" + mode:
            raise OSError(errno.ENXIO, PRIVATE)
        terminal = builtins.open(path, mode, *args, **kwargs)
        # Success and input tests use real unwrapped TextIOWrapper handles.
        return terminal if fault == "none" else Terminal(terminal, mode)
    cli.open = terminal_open
    try:
        code = cli.main(["--plan", str(fixture["path"]), "--execute"])
        result = {"exit_code": code, "stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                  "calls": len(calls), "opened": opened}
    except BaseException as exc:
        result = {"uncaught_type": type(exc).__name__, "calls": len(calls),
                  "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    os.write(result_fd, json.dumps(result).encode())


class Child:
    def __init__(self, pid, terminal_fd, result_fd, fixture):
        self.pid, self.terminal_fd, self.result_fd = pid, terminal_fd, result_fd
        self.fixture, self.output, self.reaped = fixture, b"", False

    def await_prompt(self):
        deadline = time.monotonic() + TIMEOUT
        marker = (cli.confirmation_text(self.fixture["plan"]) + "\n> ").encode()
        while marker not in self.output.replace(b"\r\n", b"\n"):
            left = deadline - time.monotonic()
            assert left > 0, "approval prompt did not appear"
            ready, _, _ = select.select([self.terminal_fd], [], [], left)
            assert ready, "approval prompt timed out"
            try:
                chunk = os.read(self.terminal_fd, 8192)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    pytest.fail("child exited before its real PTY prompt")
                raise
            assert chunk
            self.output += chunk
        rendered = self.output.replace(b"\r\n", b"\n").decode()
        assert rendered.endswith(cli.confirmation_text(self.fixture["plan"]) + "\n> ")
        assert "予約額は請求額ではありません。" in rendered

    def assert_waiting_without_activation(self):
        ready, _, _ = select.select([self.result_fd], [], [], 0.1)
        assert not ready, "approval completed before a terminal answer"
        assert os.waitpid(self.pid, os.WNOHANG) == (0, 0)
        identity = self.fixture["plan"]["identity"]
        assert not Path(identity["enabled_directory"]).exists()
        assert not Path(identity["authority_path"]).exists()

    def answer(self, data):
        os.write(self.terminal_fd, data)

    def disconnect(self):
        os.close(self.terminal_fd)
        self.terminal_fd = None

    def result(self):
        deadline, data = time.monotonic() + TIMEOUT, b""
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "fixture child result timed out"
            ready, _, _ = select.select([self.result_fd], [], [], remaining)
            assert ready, "fixture child result timed out"
            chunk = os.read(self.result_fd, 8192)
            if not chunk:
                break
            data += chunk
        while True:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.reaped = True
                assert os.waitstatus_to_exitcode(status) == 0
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert data, "fixture child produced no diagnostic result"
        result = json.loads(data)
        assert "uncaught_type" not in result, result
        return result


@contextmanager
def terminal_child(fixture, *, stdin_mode="tty", fault="none", controlling=True):
    read_fd, write_fd = os.pipe()
    if controlling:
        pid, terminal_fd = pty.fork()
    else:
        pid, terminal_fd = os.fork(), None
    if pid == 0:
        try:
            os.close(read_fd)
            if not controlling:
                os.setsid()
            child_main(fixture, write_fd, stdin_mode=stdin_mode, fault=fault)
            os.close(write_fd)
            os._exit(0)
        except BaseException:
            os._exit(127)
    os.close(write_fd)
    child = Child(pid, terminal_fd, read_fd, fixture)
    try:
        yield child
    finally:
        if not child.reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        if child.terminal_fd is not None:
            os.close(child.terminal_fd)
        os.close(read_fd)


def assert_rejected(result, code):
    assert result["exit_code"] == 5 and result["calls"] == 0
    assert result["stdout"] == ""
    assert json.loads(result["stderr"]) == {"status": "STOPPED", "code": code}
    assert PRIVATE not in result["stderr"]


def assert_twenty_once(fixture, result):
    assert result["exit_code"] == 0 and result["stderr"] == "" and result["calls"] == 20
    summary = json.loads(result["stdout"])
    assert summary["supplemental_attempted"] == 20 and summary["supplemental_not_run"] == 0
    directory = Path(fixture["plan"]["identity"]["enabled_directory"])
    approval = read_json(directory / "approval.json")
    assert approval["scope"] == runner.APPROVAL_SCOPE and approval["confirmation_channel"] == "/dev/tty"
    entries = read_json(directory / "ledger.json")["entries"]
    assert [entry["schedule_index"] for entry in entries] == list(range(940, 960))
    assert entries[0]["retry"] is True and all(entry["retry"] is False for entry in entries[1:])
    assert len(list((directory / "reservations").glob("*.json"))) == 20
    assert_originals(fixture)


@pytest.mark.parametrize("stdin_mode", ["tty", "null"])
def test_real_tty_exact_line_activates_fixture_once(fixture_plan, stdin_mode):
    exact = cli.confirmation_text(fixture_plan["plan"]).encode()
    with terminal_child(fixture_plan, stdin_mode=stdin_mode) as child:
        child.await_prompt(); child.assert_waiting_without_activation()
        child.answer(exact + b"\n")
        result = child.result()
    assert result["opened"] == [{"mode": "r", "encoding": "utf-8"}, {"mode": "w", "encoding": "utf-8"}]
    assert_twenty_once(fixture_plan, result)


@pytest.mark.parametrize("answer_kind,code", [
    ("mismatch", "supplemental_exact_approval_not_given_no_activation"),
    ("leading_space", "supplemental_exact_approval_not_given_no_activation"),
    ("trailing_space", "supplemental_exact_approval_not_given_no_activation"),
    ("blank", "supplemental_exact_approval_not_given_no_activation"),
    ("wrong_then_exact", "supplemental_exact_approval_not_given_no_activation"),
    ("eof", "supplemental_approval_eof_no_activation"),
    ("partial_eof", "supplemental_exact_approval_not_given_no_activation"),
    ("interrupt", "supplemental_approval_interrupted_no_activation"),
])
def test_real_tty_rejections_preserve_disabled_fixture(fixture_plan, answer_kind, code):
    before = tree_snapshot(fixture_plan)
    exact = cli.confirmation_text(fixture_plan["plan"]).encode()
    answer = {"mismatch": b"wrong\n", "leading_space": b" " + exact + b"\n",
              "trailing_space": exact + b" \n", "blank": b"\n",
              "wrong_then_exact": b"wrong\n" + exact + b"\n", "eof": b"\x04",
              "partial_eof": exact + b"\x04\x04", "interrupt": b"\x03"}[answer_kind]
    with terminal_child(fixture_plan) as child:
        child.await_prompt(); child.assert_waiting_without_activation()
        child.answer(answer)
        assert_rejected(child.result(), code)
    assert tree_snapshot(fixture_plan) == before


@pytest.mark.parametrize("stdin_mode", ["pipe", "environment"])
def test_stdin_or_environment_exact_text_cannot_approve(fixture_plan, stdin_mode):
    before = tree_snapshot(fixture_plan)
    with terminal_child(fixture_plan, stdin_mode=stdin_mode) as child:
        child.await_prompt(); child.assert_waiting_without_activation()
        child.answer(b"terminal refusal\n")
        assert_rejected(child.result(), "supplemental_exact_approval_not_given_no_activation")
    assert tree_snapshot(fixture_plan) == before


def test_no_controlling_tty_with_piped_approval_cannot_activate(fixture_plan):
    before = tree_snapshot(fixture_plan)
    with terminal_child(fixture_plan, stdin_mode="pipe", controlling=False) as child:
        assert_rejected(child.result(), "supplemental_tty_unavailable_no_activation")
    assert tree_snapshot(fixture_plan) == before


def test_invalid_utf8_is_a_safe_io_error(fixture_plan):
    before = tree_snapshot(fixture_plan)
    with terminal_child(fixture_plan) as child:
        child.await_prompt(); child.assert_waiting_without_activation()
        child.answer(b"\xff\n")
        assert_rejected(child.result(), "supplemental_tty_io_error_no_activation")
    assert tree_snapshot(fixture_plan) == before


def test_disconnected_controlling_tty_cannot_activate(fixture_plan):
    before = tree_snapshot(fixture_plan)
    with terminal_child(fixture_plan) as child:
        child.await_prompt(); child.assert_waiting_without_activation()
        child.disconnect()
        assert_rejected(child.result(), "supplemental_tty_io_error_no_activation")
    assert tree_snapshot(fixture_plan) == before


@pytest.mark.parametrize("fault,code", [
    ("open_r", "supplemental_tty_unavailable_no_activation"),
    ("open_w", "supplemental_tty_unavailable_no_activation"),
    ("write", "supplemental_tty_io_error_no_activation"),
    ("flush", "supplemental_tty_io_error_no_activation"),
    ("read", "supplemental_tty_io_error_no_activation"),
    ("close_r", "supplemental_tty_io_error_no_activation"),
    ("close_w", "supplemental_tty_io_error_no_activation"),
    ("read_eof", "supplemental_approval_eof_no_activation"),
    ("not_tty_r", "supplemental_real_tty_required_no_activation"),
    ("not_tty_w", "supplemental_real_tty_required_no_activation"),
])
def test_real_handle_faults_never_activate_or_expose_private_errors(fixture_plan, fault, code):
    before = tree_snapshot(fixture_plan)
    with terminal_child(fixture_plan, fault=fault) as child:
        if fault.startswith("close_"):
            child.await_prompt(); child.assert_waiting_without_activation()
            child.answer(cli.confirmation_text(fixture_plan["plan"]).encode() + b"\n")
        assert_rejected(child.result(), code)
    assert tree_snapshot(fixture_plan) == before


def test_double_launch_from_disabled_state_does_not_duplicate_fixture_calls(fixture_plan):
    exact = cli.confirmation_text(fixture_plan["plan"]).encode() + b"\n"
    with terminal_child(fixture_plan) as first:
        # Wait until the first status check has released source locks so both
        # processes reach the same disabled-state approval race deterministically.
        first.await_prompt(); first.assert_waiting_without_activation()
        with terminal_child(fixture_plan) as second:
            second.await_prompt(); second.assert_waiting_without_activation()
            first.answer(exact)
            first_result = first.result()
            assert_twenty_once(fixture_plan, first_result)
            directory = Path(fixture_plan["plan"]["identity"]["enabled_directory"])
            before_second = tree_snapshot(fixture_plan)
            second.answer(exact)
            second_result = second.result()
            assert_rejected(second_result, "supplemental_lineage_already_claimed_never_reset_or_reactivate")
    assert tree_snapshot(fixture_plan) == before_second
    assert len(read_json(directory / "ledger.json")["entries"]) == 20
    assert_originals(fixture_plan)
