"""Network-free real-PTY reproduction; never imports or invokes the launcher.

Run with the server interpreter and redirect stdout to tty_reproduction.json.
The child opens the controlling TTY only; it never reads or sends approval text.
"""
from __future__ import annotations

import datetime
import json
import os
import pty
import select
import signal
import sys
import time


def child_probe() -> dict:
    result = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "stdin_is_tty": os.isatty(0),
        "has_controlling_tty": os.tcgetpgrp(0) == os.getpgrp(),
        "research_api_sends": 0,
        "launcher_invoked": False,
        "approval_input_sent": False,
    }
    try:
        with open("/dev/tty", "r+", encoding="utf-8"):
            pass
        result["original_r_plus"] = {"opened": True}
    except Exception as exc:
        # The operation is fixed and has no user input or secret values.
        result["original_r_plus"] = {
            "opened": False,
            "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
            "exception_message": str(exc),
            "errno": getattr(exc, "errno", None),
            "is_oserror": isinstance(exc, OSError),
        }
    try:
        with open("/dev/tty", "r", encoding="utf-8") as reader:
            with open("/dev/tty", "w", encoding="utf-8", buffering=1) as writer:
                result["separate_read_write"] = {
                    "opened": True,
                    "reader_is_tty": reader.isatty(),
                    "writer_is_tty": writer.isatty(),
                }
    except Exception as exc:
        result["separate_read_write"] = {
            "opened": False,
            "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
            "exception_message": str(exc),
            "errno": getattr(exc, "errno", None),
        }
    return result


def main() -> int:
    read_fd, write_fd = os.pipe()
    pid, master_fd = pty.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            payload = json.dumps(child_probe(), sort_keys=True).encode("utf-8")
            os.write(write_fd, payload)
            os.close(write_fd)
            os._exit(0)
        except BaseException:
            os._exit(1)
    os.close(write_fd)
    deadline = time.monotonic() + 5.0
    chunks = []
    status = None
    timed_out = False
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([read_fd], [], [], 0.05)
            if readable:
                chunk = os.read(read_fd, 65536)
                if chunk:
                    chunks.append(chunk)
            waited, value = os.waitpid(pid, os.WNOHANG)
            if waited:
                status = value
                break
        if status is None:
            timed_out = True
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
        # Child has exited, so reading remaining pipe bytes cannot block.
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        if status is None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
        os.close(read_fd)
        os.close(master_fd)
    result = {
        "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "timeout_seconds": 5,
        "timed_out": timed_out,
        "child_reaped": True,
        "child_exit_code": os.waitstatus_to_exitcode(status),
        "probe": json.loads(b"".join(chunks)) if chunks else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if timed_out or result["child_exit_code"] != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
