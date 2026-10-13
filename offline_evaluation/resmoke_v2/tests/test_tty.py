"""Real controlling-terminal regression tests; only fixture approval is entered.

Children call the confirmation function directly. They never load a real plan,
invoke the launcher, create an enabled run, or use a research transport.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time

import pytest


FIXTURE_APPROVAL = "TTY_FIXTURE_APPROVAL_ONLY"
REPO_ROOT = Path(__file__).resolve().parents[3]
TIMEOUT = 5.0

# A separate result pipe makes waiting for input distinguishable from terminal
# echo. stdin can be replaced without changing the child's controlling terminal.
CHILD = r'''
import builtins
import errno
import json
import os
import sys
from offline_evaluation.resmoke_v2 import cli

result_fd = int(sys.argv[1])
stdin_mode, fault = sys.argv[2:4]
if stdin_mode == "null":
    fd = os.open("/dev/null", os.O_RDONLY)
    os.dup2(fd, 0)
    os.close(fd)
elif stdin_mode == "pipe":
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"TTY_FIXTURE_APPROVAL_ONLY\n")
    os.close(write_fd)
    os.dup2(read_fd, 0)
    os.close(read_fd)

original_open = builtins.open
class FaultyTerminal:
    def __init__(self, terminal):
        self.terminal = terminal
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.terminal.close()
    def __getattr__(self, name):
        return getattr(self.terminal, name)
    def write(self, value):
        if fault == "write":
            raise OSError(errno.EIO, "PRIVATE_FIXTURE_ERROR_NEVER_REPORT")
        return self.terminal.write(value)
    def flush(self):
        if fault == "flush":
            raise OSError(errno.EIO, "PRIVATE_FIXTURE_ERROR_NEVER_REPORT")
        return self.terminal.flush()
    def readline(self):
        if fault == "read":
            raise OSError(errno.EIO, "PRIVATE_FIXTURE_ERROR_NEVER_REPORT")
        return self.terminal.readline()

if fault != "none":
    def faulty_open(path, mode="r", *args, **kwargs):
        if path == "/dev/tty" and fault == "open_" + mode:
            raise OSError(errno.ENXIO, "PRIVATE_FIXTURE_ERROR_NEVER_REPORT")
        terminal = original_open(path, mode, *args, **kwargs)
        return FaultyTerminal(terminal) if path == "/dev/tty" else terminal
    builtins.open = faulty_open

try:
    approved = cli._confirm_from_tty("TTY_FIXTURE_APPROVAL_ONLY")
    result = {"approved": approved}
except Exception as exc:
    result = {"error_type": type(exc).__name__, "error": str(exc)}
os.write(result_fd, json.dumps(result).encode("utf-8"))
'''


class TerminalChild:
    def __init__(self, pid, terminal_fd, result_fd):
        self.pid = pid
        self.terminal_fd = terminal_fd
        self.result_fd = result_fd
        self.output = b""
        self.reaped = False

    def await_prompt(self):
        deadline = time.monotonic() + TIMEOUT
        marker = f"Type exactly: {FIXTURE_APPROVAL}".encode()
        while marker not in self.output or b"> " not in self.output:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "confirmation prompt did not appear before timeout"
            ready, _, _ = select.select([self.terminal_fd], [], [], remaining)
            assert ready, "confirmation prompt did not appear before timeout"
            try:
                chunk = os.read(self.terminal_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    pytest.fail("confirmation child exited before displaying prompt")
                raise
            assert chunk, "confirmation child closed terminal before prompt"
            self.output += chunk

    def assert_waiting(self):
        ready, _, _ = select.select([self.result_fd], [], [], 0.1)
        assert not ready, "confirmation completed before receiving terminal input"
        assert os.waitpid(self.pid, os.WNOHANG) == (0, 0)

    def answer(self, value):
        os.write(self.terminal_fd, value)

    def result(self):
        deadline = time.monotonic() + TIMEOUT
        data = b""
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "confirmation child did not finish before timeout"
            ready, _, _ = select.select([self.result_fd], [], [], remaining)
            assert ready, "confirmation child did not finish before timeout"
            chunk = os.read(self.result_fd, 4096)
            if not chunk:
                break
            data += chunk
        # EOF on the dedicated pipe comes when the child exits; still bound the
        # reap rather than relying on that assumption for process cleanup.
        while True:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.reaped = True
                assert os.waitstatus_to_exitcode(status) == 0
                break
            assert time.monotonic() < deadline, "confirmation child failed to exit"
            time.sleep(0.01)
        assert data, "confirmation child produced no result"
        return json.loads(data)


@contextmanager
def controlling_terminal(*, stdin_mode="tty", fault="none"):
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    pid, terminal_fd = pty.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            os.chdir(REPO_ROOT)
            os.execv(sys.executable, [sys.executable, "-B", "-c", CHILD,
                str(write_fd), stdin_mode, fault])
        finally:
            os._exit(127)
    os.close(write_fd)
    child = TerminalChild(pid, terminal_fd, read_fd)
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
        os.close(terminal_fd)
        os.close(read_fd)


@pytest.mark.parametrize("answer,approved", [
    (FIXTURE_APPROVAL.encode() + b"\n", True),
    (b"different fixture answer\n", False),
    (FIXTURE_APPROVAL.encode() + b" \n", False),
    (b"wrong first line\n" + FIXTURE_APPROVAL.encode() + b"\n", False),
    (b"\n", False),
    (b"\x04", False),  # canonical terminal EOF
    (FIXTURE_APPROVAL.encode() + b"\x04\x04", False),  # partial line then EOF
    (b"\x03", False),  # terminal-generated SIGINT / KeyboardInterrupt
])
def test_real_tty_prompt_waits_and_reads_exact_fixture_line(answer, approved):
    with controlling_terminal() as child:
        child.await_prompt()
        child.assert_waiting()
        child.answer(answer)
        assert child.result() == {"approved": approved}


def test_invalid_utf8_terminal_input_is_a_safe_io_failure():
    with controlling_terminal() as child:
        child.await_prompt()
        child.assert_waiting()
        child.answer(b"\xff\n")
        result = child.result()
    assert result == {
        "error_type": "TTYConfirmationError",
        "error": "TTY_IO_ERROR: error_type=UnicodeDecodeError; errno=None",
    }


@pytest.mark.parametrize("stdin_mode,answer,approved", [
    ("null", FIXTURE_APPROVAL.encode() + b"\n", True),
    # The pipe itself contains the exact fixture approval. Only the differing
    # answer from /dev/tty may affect the result.
    ("pipe", b"terminal refusal\n", False),
])
def test_real_tty_used_even_when_stdin_is_redirected(stdin_mode, answer, approved):
    with controlling_terminal(stdin_mode=stdin_mode) as child:
        child.await_prompt()
        child.assert_waiting()
        child.answer(answer)
        assert child.result() == {"approved": approved}


def test_no_controlling_terminal_is_unavailable_even_with_stdin_approval():
    completed = subprocess.run(
        [sys.executable, "-B", "-c", CHILD, "1", "tty", "none"],
        input=FIXTURE_APPROVAL.encode() + b"\n", capture_output=True,
        cwd=REPO_ROOT, start_new_session=True, timeout=TIMEOUT, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["error_type"] == "TTYConfirmationError"
    assert result["error"].startswith("TTY_UNAVAILABLE:")
    assert "error_type=OSError" in result["error"]
    assert f"errno={errno.ENXIO}" in result["error"]
    assert not completed.stderr


@pytest.mark.parametrize("fault,stop_reason,number", [
    ("open_r", "TTY_UNAVAILABLE", errno.ENXIO),
    ("open_w", "TTY_UNAVAILABLE", errno.ENXIO),
    ("write", "TTY_IO_ERROR", errno.EIO),
    ("flush", "TTY_IO_ERROR", errno.EIO),
    ("read", "TTY_IO_ERROR", errno.EIO),
])
def test_tty_failures_are_distinct_from_refusal_without_private_error_text(fault, stop_reason, number):
    # Fault injection wraps handles obtained from a real controlling terminal;
    # the success/input paths above run without any open() mocks.
    with controlling_terminal(fault=fault) as child:
        result = child.result()
    assert result["error_type"] == "TTYConfirmationError"
    assert result["error"].startswith(stop_reason + ":")
    assert "error_type=OSError" in result["error"]
    assert f"errno={number}" in result["error"]
    assert "approved" not in result
    assert "PRIVATE_FIXTURE" not in result["error"]
    assert FIXTURE_APPROVAL not in result["error"]
