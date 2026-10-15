"""Offline CLI gates: preview is read-only and payment activation needs exact TTY consent."""
import io
import json
from pathlib import Path

import pytest

from offline_evaluation.operations.goal006b import cli, runner
from offline_evaluation.operations.goal006a.archive import RecoveryError


def tree_snapshot(fixture):
    root = fixture["root"].parent
    return {p.relative_to(root).as_posix(): ("directory", None) if p.is_dir() else ("file", p.read_bytes())
            for p in root.rglob("*")}


def forbid_activation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("CLI must not activate or enter the generation runner")
    monkeypatch.setattr(runner, "initialize_enabled", forbidden)
    monkeypatch.setattr(runner, "run", forbidden)


def test_default_preview_creates_no_files_and_never_activates(fixture_plan, monkeypatch, capsys):
    forbid_activation(monkeypatch)
    before = tree_snapshot(fixture_plan)
    assert cli.main(["--plan", str(fixture_plan["path"])]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["approved"] is False and result["api_allowed"] is False
    assert result["preview_only"] is True and result["supplemental_not_run"] == 20
    assert result["research_api_requests_sent_by_preview"] == 0
    assert result["required_new_approval_scope"] == runner.APPROVAL_SCOPE
    assert tree_snapshot(fixture_plan) == before


def test_disabled_checkpoint_creates_no_files_including_locks(fixture_plan, monkeypatch, capsys):
    forbid_activation(monkeypatch)
    monkeypatch.setattr(runner, "_save", lambda *_a: pytest.fail("disabled checkpoint must not save"))
    before = tree_snapshot(fixture_plan)
    assert cli.main(["--plan", str(fixture_plan["path"]), "--checkpoint"]) == 5
    result = json.loads(capsys.readouterr().err)
    assert result == {"status": "STOPPED", "code": "supplement_not_enabled_checkpoint_cannot_create_state"}
    assert tree_snapshot(fixture_plan) == before


@pytest.mark.parametrize("tty_case", ["missing", "not_tty", "wrong_confirmation"])
def test_execute_without_exact_real_tty_consent_never_activates(fixture_plan, monkeypatch, capsys, tty_case):
    forbid_activation(monkeypatch)
    opened = []
    class Terminal(io.StringIO):
        def isatty(self):
            return tty_case != "not_tty"
        def readline(self, *_args):
            return "APPROVE\n"

    def fake_open(path, mode):
        assert path == "/dev/tty" and mode == "r+"
        opened.append(path)
        if tty_case == "missing":
            raise OSError("fixture has no controlling terminal")
        return Terminal()
    monkeypatch.setattr(cli, "open", fake_open, raising=False)
    before = tree_snapshot(fixture_plan)
    assert cli.main(["--plan", str(fixture_plan["path"]), "--execute"]) == 5
    result = json.loads(capsys.readouterr().err)
    expected = {"missing": "supplemental_real_tty_required_no_activation",
                "not_tty": "supplemental_real_tty_required",
                "wrong_confirmation": "supplemental_exact_approval_not_given_no_activation"}
    assert result == {"status": "STOPPED", "code": expected[tty_case]}
    assert opened == ["/dev/tty"]
    assert tree_snapshot(fixture_plan) == before
    assert not Path(fixture_plan["plan"]["identity"]["enabled_directory"]).exists()


def test_recovery_error_prints_only_safe_structured_diagnostics(fixture_plan, monkeypatch, capsys):
    def fail(*_args):
        try:
            raise RuntimeError("fixture-secret-token signed=https://private.invalid/?token=fixture-secret-token")
        except RuntimeError as cause:
            raise RecoveryError("asset_download", "github_command_failed", returncode=1, http_status=503) from cause
    monkeypatch.setattr(runner, "status", fail)
    forbid_activation(monkeypatch)
    before = tree_snapshot(fixture_plan)
    assert cli.main(["--plan", str(fixture_plan["path"])]) == 5
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"status": "STOPPED", "error": {
        "stage": "asset_download", "code": "github_command_failed", "returncode": 1, "http_status": 503}}
    assert "fixture-secret-token" not in captured.err and "private.invalid" not in captured.err
    assert tree_snapshot(fixture_plan) == before
