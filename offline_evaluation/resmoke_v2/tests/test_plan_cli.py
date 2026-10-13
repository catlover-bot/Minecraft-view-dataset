"""Input isolation and the launcher's zero-send refusal paths."""
import pytest

from offline_evaluation.server_v2.generation import _load_run, digest, object_hash
from offline_evaluation.resmoke_v2 import cli, cuboids, plan


def test_ordinary_bodies_do_not_access_or_include_exact_gt(monkeypatch):
    records = []
    for scene in plan.SCENES:
        for condition in "OJPE":
            text = f"saved_{scene}_{condition}"
            records.append({"scene_id": scene, "condition": condition,
                "input_text": text, "input_sha256": digest(text.encode()),
                "input_id": object_hash([scene, condition]), "source_sha256": "a" * 64})
    accesses = []
    class GTOnlyForE:
        def __getitem__(self, scene):
            accesses.append(scene)
            return "GT_COORDINATE_SENTINEL"
    monkeypatch.setattr(cuboids, "encode_exact", lambda grid: {"sentinel": grid})
    monkeypatch.setattr(cuboids, "exact_builder_text", lambda spec: spec["sentinel"])
    new = plan.make_inputs(records, GTOnlyForE())
    assert accesses == list(plan.SCENES)
    for before, after in zip(records, new):
        if after["condition"] in "OJP":
            assert after["input_text"] == before["input_text"]
            assert "GT_COORDINATE_SENTINEL" not in str(after)
        else:
            assert after["input_text"] == "GT_COORDINATE_SENTINEL"
        assert after["input_id"] != before["input_id"]
        assert after["description_model"] == ("openai" if after["condition"] in "OJ" else None)


def test_original_paid_campaign_still_readable():
    # Existing hash-bound validators remain the authority; this is read-only.
    prior = plan.parent_snapshot()
    assert prior["usage"] == plan.PARENT_USAGE
    for provider in plan.MODELS:
        manifest, config, records, _, state = _load_run(plan.PARENT_DIRECTORY / "runs" / provider)
        assert len(records) == 8
        assert manifest["identity"]["implementation_hashes"]
        assert config.model == plan.MODELS[provider]
        assert len(state["attempts"]) == (11 if provider == "openai" else 3)


def test_fresh_pricing_uses_existing_validator_without_zero_age_configuration():
    import datetime
    from offline_evaluation.server_v2.generation import read_json
    source = read_json(plan.PRICING_PATH)
    pricing = {p: source[p] for p in plan.MODELS}
    today = datetime.datetime.now(datetime.timezone.utc).date()
    for record in pricing.values():
        record["verified_on"] = today.isoformat()
    plan.validate_pricing(pricing, fresh=True)
    pricing["openai"]["verified_on"] = (today - datetime.timedelta(days=1)).isoformat()
    with pytest.raises(RuntimeError, match="today_verification"):
        plan.validate_pricing(pricing, fresh=True)


@pytest.mark.parametrize("field", ["voxels_sha256", "bbox_sha256"])
def test_new_plan_rejects_gt_drift_from_original_smoke(field):
    record = {"scene_id": "fixed", "condition": "E", "source_sha256": "v", "source_GT_bbox_sha256": "b"}
    descriptor = {"voxels_sha256": "v", "bbox_sha256": "b"}
    plan.verify_gt_sources([record], {"fixed": descriptor})
    descriptor[field] = "changed"
    with pytest.raises(RuntimeError, match="frozen old smoke"):
        plan.verify_gt_sources([record], {"fixed": descriptor})


class Terminal:
    def __init__(self, answer):
        self.answer = answer
        self.output = ""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def write(self, value):
        self.output += value
    def flush(self):
        pass
    def readline(self):
        return self.answer


@pytest.mark.parametrize("answer", ["", "\n", "yes\n", "APPROVE SOMETHING ELSE\n"])
def test_tty_enter_eof_mismatch_never_approve(monkeypatch, answer):
    terminal = Terminal(answer)
    def opener(path, mode, **kwargs):
        assert path == "/dev/tty" and mode == "r+"
        return terminal
    monkeypatch.setattr("builtins.open", opener)
    assert cli._confirm_from_tty("EXACT_REQUIRED_STRING") is False
    assert "EXACT_REQUIRED_STRING" in terminal.output


def test_missing_tty_refuses(monkeypatch):
    def absent(*args, **kwargs):
        raise OSError("no terminal")
    monkeypatch.setattr("builtins.open", absent)
    assert cli._confirm_from_tty("EXACT_REQUIRED_STRING") is False


@pytest.mark.parametrize("preview_only,keys,answer,expected", [
    (True, True, True, 0), (False, False, True, 3), (False, True, False, 2),
])
def test_launcher_non_send_paths(monkeypatch, tmp_path, capsys, preview_only, keys, answer, expected):
    from offline_evaluation.resmoke_v2 import runner
    fake = {"identity_hash": "b" * 64, "identity": {
        "parent": {}, "pricing": {}, "enabled_directory": str(tmp_path / "never_enabled")}}
    monkeypatch.setattr(plan, "load_plan", lambda _: fake)
    monkeypatch.setattr(plan, "verify_parent", lambda _: None)
    monkeypatch.setattr(plan, "validate_pricing", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "preview", lambda _: {"api_key_present": {"openai": keys, "anthropic": keys}})
    calls = []
    def approval(expected):
        calls.append("tty")
        return answer
    monkeypatch.setattr(cli, "_confirm_from_tty", approval)
    def forbidden(*args, **kwargs):
        pytest.fail("zero-send path reached activation or research transport")
    monkeypatch.setattr(runner, "initialize_enabled", forbidden)
    monkeypatch.setattr(runner, "run", forbidden)
    assert cli.launch("unused", preview_only=preview_only) == expected
    assert calls == ([] if preview_only or not keys else ["tty"])
    assert not (tmp_path / "never_enabled").exists()


def test_key_display_is_presence_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "DO_NOT_LOG_KEY_SENTINEL")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DO_NOT_LOG_KEY_SENTINEL")
    fake = {"status": "disabled", "identity_hash": "b" * 64, "identity": {
        "protocol_id": plan.PROTOCOL_ID, "schedule": [], "schema_sha256": "s", "prompt_sha256": "p",
        "implementation_hashes": {}, "budgets": {}, "parent": {"usage": {}, "ledger_sha256": "l"},
        "reservation_policy": "r", "pricing": {p: {"verified_on": "2026-10-13"} for p in plan.MODELS},
        "enabled_directory": "unused", "authority_path": "unused"}}
    display = cli.preview(fake)
    assert display["api_key_present"] == {"openai": True, "anthropic": True}
    assert "DO_NOT_LOG_KEY_SENTINEL" not in str(display)


@pytest.mark.parametrize("relative", [".env", ".ssh/key", ".git/config", "../outside", "/root/.codex/auth.json"])
def test_reused_publication_boundary_rejects_sensitive_paths(relative):
    from offline_evaluation.server_v2.checkpoint import safe_relative
    with pytest.raises(ValueError):
        safe_relative(relative)


def test_publication_scanner_suppresses_secret_value(tmp_path):
    from offline_evaluation.server_v2.checkpoint import scan_file
    target = tmp_path / "response.json"
    marker = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    target.write_text(marker)
    with pytest.raises(ValueError, match="value suppressed") as raised:
        scan_file(target)
    assert marker not in str(raised.value)
