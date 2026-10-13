"""Durable scheduling/accounting tests. Every transport is a CPU fixture."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path

import pytest

from offline_evaluation.resmoke_v2 import adapter, runner
from offline_evaluation.server_v2.core import Grid, grid_to_payload
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json
from offline_evaluation.server_v2 import providers


STONE = {"operations": [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"}]}
PRIOR = {"calls": 14, "retries": 4, "tokens": 173509,
         "cost_usd_upper_bound": 0.25857425, "unknown_usage_attempts": 0}


def response(provider, text=None, **changes):
    text = json.dumps(STONE) if text is None else text
    value = {"id": "fixture_response", "model": adapter.MODELS[provider],
             "usage": {"input_tokens": 10, "output_tokens": 20}}
    if provider == "openai":
        value.update(model="gpt-5-mini-2025-08-07", status="completed",
                     output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}])
    else:
        value.update(stop_reason="end_turn", content=[{"type": "text", "text": text}])
    value.update(changes)
    return json.dumps(value).encode()


@pytest.fixture
def fixture_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture_only_never_live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture_only_never_live")
    monkeypatch.setattr(adapter, "http_transport", lambda *_: pytest.fail("live transport forbidden"))
    monkeypatch.setattr(providers, "http_transport", lambda *_: pytest.fail("live transport forbidden"))
    parent = tmp_path / "parent"
    parent.mkdir()
    write_json(parent / "campaign.json", {"identity": {"runs": {
        p: {"path": "runs/" + p} for p in adapter.MODELS}}})
    write_json(parent / "ledger.json", {"fixture_parent_usage": PRIOR})
    for provider in adapter.MODELS:
        (parent / "runs" / provider).mkdir(parents=True)
        (parent / "runs" / provider / "run.lock").write_bytes(b"immutable parent lock")
        write_json(parent / "runs" / provider / "state.json", {"fixture_state": provider})
    for name in ("run.lock", "budget.lock"):
        (parent / name).write_bytes(b"immutable parent lock")
    before = {p.relative_to(parent).as_posix(): p.read_bytes() for p in parent.rglob("*") if p.is_file()}
    parent_hashes = {name: runner.digest(data) for name, data in before.items()}
    def verify(identity):
        directory = Path(identity["parent"]["campaign_directory"])
        actual = {p.relative_to(directory).as_posix(): runner.digest(p.read_bytes())
                  for p in directory.rglob("*") if p.is_file()}
        if actual != parent_hashes:
            raise GenerationError("parent_changed; replan required")
    monkeypatch.setattr(runner, "_verify_parent", verify)
    pricing = {}
    for provider, model in adapter.MODELS.items():
        pricing[provider] = {"provider": provider, "model": model, "verified": True,
            "verified_on": runner.utc_now()[:10],
            "context_window_tokens": 400000 if provider == "openai" else 200000,
            "input_usd_per_million": .25 if provider == "openai" else 1,
            "output_usd_per_million": 2 if provider == "openai" else 5,
            "accepted_response_model_ids": [model] + (["gpt-5-mini-2025-08-07"] if provider == "openai" else [])}
    gt = Grid({(0, 0, 0): "stone"})
    scenes = ("fixture_building_060", "fixture_building_075")
    schedule = [{"input_id": scene + condition, "scene_id": scene, "condition": condition,
                 "description_model": "openai" if condition in "OJ" else None,
                 "provider": provider, "model": adapter.MODELS[provider],
                 "input_text": scene + " " + condition}
                for scene in scenes for condition in "OJPE" for provider in adapter.MODELS]
    identity = {"protocol_id": "fixture_goal004", "provenance": "fixture_only",
                "enabled_directory": str(tmp_path / "enabled"), "authority_path": str(tmp_path / "authority.json"),
                "parent": {"campaign_directory": str(parent), "usage": dict(PRIOR)},
                "schedule": schedule, "prompt": "Fixture common prompt.", "pricing": pricing,
                "budgets": {"max_new_calls": 16, "max_total_calls": 30, "max_new_retries": 0,
                            "max_cost_usd": 5, "max_tokens": 5324288, "max_total_tokens": 5497797,
                            "max_output_tokens": 32768, "concurrency": 1},
                "gt_by_scene": {s: {"grid_sha256": object_hash(grid_to_payload(gt))} for s in scenes}}
    plan = {"identity": identity, "identity_hash": object_hash(identity), "api_allowed": False}
    path = tmp_path / "plan.json"
    write_json(path, plan)
    def load(path):
        value = read_json(Path(path))
        if object_hash(value["identity"]) != value["identity_hash"]:
            raise GenerationError("fixture_plan_hash_mismatch")
        return value
    monkeypatch.setattr(runner, "_plan", load)
    return {"path": path, "plan": plan, "before": before, "parent": parent,
            "grids": {s: gt for s in scenes}}


def approval(plan):
    return {"manual_confirmed": True, "plan_identity_hash": plan["identity_hash"],
            "enabled_directory": plan["identity"]["enabled_directory"],
            "approved_at_utc": runner.utc_now(), "confirmation_channel": "/dev/tty",
            "scope": runner.APPROVAL_SCOPE}


def enable(fixture):
    return runner.initialize_enabled(fixture["path"], approval=approval(fixture["plan"]))


def transports(calls, handler=None):
    def factory(provider):
        def send(request, timeout):
            payload = json.loads(request.data)
            calls.append((provider, payload))
            return handler(provider, payload, len(calls)) if handler else response(provider)
        return send
    return {p: factory(p) for p in adapter.MODELS}


def test_disabled_and_permission_gates_send_zero(fixture_plan):
    calls = []
    with pytest.raises(GenerationError, match="permission"):
        runner.run(fixture_plan["path"], transports=transports(calls))
    with pytest.raises(GenerationError, match="not_enabled"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    with pytest.raises(GenerationError, match="fresh_exact_scope"):
        runner.initialize_enabled(fixture_plan["path"], approval={"manual_confirmed": True})
    assert not calls


def test_fixed16_interleaving_reservations_resume_and_snapshot_evaluation(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def inspect(provider, payload, count):
        ledger = read_json(directory / "ledger.json")
        assert len(ledger["entries"]) == count
        assert ledger["entries"][-1]["status"] == "reserved"
        assert ledger["entries"][-1]["usage"] is None
        assert ledger["entries"][-1]["charged_output_tokens"] == 32768
        return response(provider)
    wire = transports(calls, inspect)
    result = runner.run(fixture_plan["path"], permission=True, transports=wire)
    assert result["experiment_complete"] is True
    assert result["used"]["combined"]["calls"] == 30
    assert result["used"]["combined"]["retries"] == 4
    assert result["used"]["combined"]["tokens"] == PRIOR["tokens"] + 480
    assert result["used"]["combined"]["cost_usd_upper_bound"] == pytest.approx(.25979425)
    assert [p for p, _ in calls] == ["openai", "anthropic"] * 8
    assert runner.run(fixture_plan["path"], permission=True, transports=wire) == result
    assert len(calls) == 16
    summary = runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])
    assert summary["stage_counts"]["completed_request_outcome"] == 16
    assert summary["stage_counts"]["valid_nonempty_reconstruction"] == 16
    assert summary["account_native_format_acceptance"] == "FIXTURE_ONLY"
    assert summary["families"] == []
    assert summary["smoke_inference_suppressed"] is True
    assert all(g["valid_rows"] == 2 for g in summary["groups"])
    after = {p.relative_to(fixture_plan["parent"]).as_posix(): p.read_bytes()
             for p in fixture_plan["parent"].rglob("*") if p.is_file()}
    assert after == fixture_plan["before"]


@pytest.mark.parametrize("text,kind,stage", [
    ('```json\n{"operations":[]}\n```', "parse_failure", "json"),
    ('{"operations":[],"operations":[]}', "parse_failure", "json"),
    ('[]', "validation_failure", "schema"),
    ('{"operations":[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]}', "validation_failure", "schema"),
    ('{"operations":[{"op":"carve","x1":0,"y1":0,"z1":0,"x2":0,"y2":0,"z2":0,"block":"air"}]}', "validation_failure", "schema"),
    ('{"operations":[{"op":"set","x":0,"y":0,"z":0,"block":"STONE"}]}', "validation_failure", "local_validator"),
])
def test_known_usage_output_failure_continues_without_retry(fixture_plan, text, kind, stage):
    directory = enable(fixture_plan)
    calls = []
    result = runner.run(fixture_plan["path"], permission=True, transports=transports(
        calls, lambda provider, _, count: response(provider, text if count == 1 else None)))
    assert result["experiment_complete"] is True
    assert len(calls) == 16
    entry = read_json(directory / "ledger.json")["entries"][0]
    assert entry["failure_kind"] == kind
    assert entry["failure_stage"] == stage
    assert all(not e["retry"] for e in read_json(directory / "ledger.json")["entries"])


@pytest.mark.parametrize("mode", ["truncation", "refusal"])
def test_known_usage_truncation_or_refusal_continues(fixture_plan, mode):
    directory = enable(fixture_plan)
    calls = []
    def handler(provider, _, count):
        if count != 1:
            return response(provider)
        if mode == "truncation":
            return response(provider, status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return response(provider, output=[{"type": "message", "content": [{"type": "refusal", "refusal": "fixture"}]}])
    result = runner.run(fixture_plan["path"], permission=True, transports=transports(calls, handler))
    assert result["experiment_complete"] is True
    assert len(calls) == 16
    assert read_json(directory / "ledger.json")["entries"][0]["failure_kind"] == mode


@pytest.mark.parametrize("text", [json.dumps({"operations": []}), json.dumps({"operations": [
    {"op": "carve", "x1": 0, "y1": 0, "z1": 0, "x2": 1, "y2": 1, "z2": 1}]})])
def test_empty_execution_is_measured_zero_but_not_nonempty_success(fixture_plan, text):
    directory = enable(fixture_plan)
    calls = []
    runner.run(fixture_plan["path"], permission=True, transports=transports(
        calls, lambda p, _, count: response(p, text if count == 1 else None)))
    summary = runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])
    assert summary["stage_counts"]["executor_success"] == 16
    assert summary["stage_counts"]["empty_output"] == 1
    assert summary["stage_counts"]["valid_nonempty_reconstruction"] == 15
    row = read_json(directory / "evaluation.json")["rows"][0]
    assert row["metrics"]["aligned"]["occupancy_iou"] == 0
    assert row["metrics"]["aligned"]["dimension_mae"] == 1
    assert row["metrics"]["aligned"]["total_normalized_repair_cost"] == 1


@pytest.mark.parametrize("kind", ["timeout", "auth_failure", "billing_failure", "model_unavailable", "schema_unsupported"])
def test_unknown_http_usage_keeps_full_reservation_and_halts_both_providers(fixture_plan, kind):
    directory = enable(fixture_plan)
    calls = []
    def handler(*_):
        raise providers.ProviderError(kind)
    wire = transports(calls, handler)
    result = runner.run(fixture_plan["path"], permission=True, transports=wire)
    assert len(calls) == 1 and not result["experiment_complete"]
    assert result["used"]["new"]["unknown_usage_attempts"] == 1
    assert result["used"]["new"]["cost_usd_upper_bound"] == .165536
    assert result["used"]["combined"]["cost_usd_upper_bound"] == .42411025
    runner.run(fixture_plan["path"], permission=True, transports=wire)
    assert len(calls) == 1
    summary = runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])
    assert summary["quality_evaluated_inputs"] == 0
    assert all(g["all_attempted_rows"] == 0 for g in summary["groups"])
    assert all(r["metrics"] is None for r in read_json(directory / "evaluation.json")["rows"])


def test_known_usage_wrong_snapshot_is_critical_and_not_quality_zero(fixture_plan):
    enable(fixture_plan)
    calls = []
    result = runner.run(fixture_plan["path"], permission=True, transports=transports(
        calls, lambda p, *_: response(p, model="not-approved-snapshot")))
    assert len(calls) == 1
    assert result["halt_reason"] == "response_identity_failure"
    summary = runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])
    assert summary["stage_counts"]["completed_request_outcome"] == 0
    assert all(g["all_attempted_rows"] == 0 for g in summary["groups"])


def test_crash_during_send_never_automatically_resends(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def crash(provider, _, count):
        if count == 5:
            raise KeyboardInterrupt
        return response(provider)
    wire = transports(calls, crash)
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan["path"], permission=True, transports=wire)
    assert read_json(directory / "ledger.json")["entries"][-1]["status"] == "reserved"
    result = runner.run(fixture_plan["path"], permission=True, transports=wire)
    assert len(calls) == 5
    assert result["halt_reason"] == "unreconciled_prior_reservation"
    assert result["used"]["new"]["unknown_usage_attempts"] == 1


def test_interruption_after_settlement_resumes_next_unattempted_only(fixture_plan, monkeypatch):
    directory = enable(fixture_plan)
    calls = []
    original = runner.write_json
    def interrupt(path, data):
        original(path, data)
        if (path == directory / "ledger.json" and len(data["entries"]) == 4
                and data["entries"][-1]["status"] == "success"):
            raise KeyboardInterrupt
    monkeypatch.setattr(runner, "write_json", interrupt)
    wire = transports(calls)
    with pytest.raises(KeyboardInterrupt):
        runner.run(fixture_plan["path"], permission=True, transports=wire)
    monkeypatch.setattr(runner, "write_json", original)
    result = runner.run(fixture_plan["path"], permission=True, transports=wire)
    assert len(calls) == 16 and result["experiment_complete"]


def test_parent_change_after_approval_prevents_any_send(fixture_plan):
    enable(fixture_plan)
    (fixture_plan["parent"] / "ledger.json").write_text("changed parent ledger")
    calls = []
    with pytest.raises(GenerationError, match="parent_changed"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert not calls


def test_parent_lock_is_held_during_request_and_cannot_run_concurrently(fixture_plan):
    enable(fixture_plan)
    path = fixture_plan["parent"] / "runs/openai/run.lock"
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(GenerationError, match="parent_generation_locked"):
            runner.run(fixture_plan["path"], permission=True, transports=transports([]))
        fcntl.flock(stream, fcntl.LOCK_UN)
    def inspect(provider, *_):
        with path.open("rb") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return response(provider)
    assert runner.run(fixture_plan["path"], permission=True,
                      transports=transports([], inspect))["experiment_complete"]


def test_combined_five_dollar_cap_includes_parent_and_uses_decimal(fixture_plan):
    identity = deepcopy(fixture_plan["plan"]["identity"])
    identity["parent"]["usage"]["cost_usd_upper_bound"] = 4.90
    assert runner._can_reserve(identity, {"entries": []}, (400000, 32768, .165536)) is False
    identity["parent"]["usage"]["cost_usd_upper_bound"] = 4.834464
    assert runner._can_reserve(identity, {"entries": []}, (400000, 32768, .165536)) is True


def test_provider_only_or_separate_run_cannot_reset_budget(fixture_plan):
    enable(fixture_plan)
    with pytest.raises(GenerationError, match="both_fixture"):
        runner.run(fixture_plan["path"], permission=True, transports={"openai": lambda *_: None})
    new = deepcopy(fixture_plan["plan"])
    new["identity"]["enabled_directory"] += "-second"
    new["identity_hash"] = object_hash(new["identity"])
    new_path = fixture_plan["path"].with_name("second-plan.json")
    write_json(new_path, new)
    with pytest.raises(GenerationError, match="lineage_already_claimed"):
        runner.initialize_enabled(new_path, approval=approval(new))
    assert not Path(new["identity"]["enabled_directory"]).exists()


def test_unknown_response_usage_has_no_zero_quality(fixture_plan):
    enable(fixture_plan)
    calls = []
    result = runner.run(fixture_plan["path"], permission=True, transports=transports(
        calls, lambda p, *_: response(p, usage=None)))
    assert len(calls) == 1
    assert result["halt_reason"] == "unknown_usage:unknown_usage"
    summary = runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])
    assert summary["stage_counts"]["api_response_received"] == 1
    assert summary["stage_counts"]["completed_request_outcome"] == 0
    assert summary["stage_counts"]["empty_output"] == 0
    assert summary["quality_evaluated_inputs"] == 0


def test_missing_one_key_sends_zero_and_spends_no_reservation(fixture_plan, monkeypatch):
    directory = enable(fixture_plan)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    calls = []
    with pytest.raises(GenerationError, match="both_provider_keys"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert not calls and read_json(directory / "ledger.json")["entries"] == []


def test_ledger_rollback_cannot_resend_immutable_claimed_inputs(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    ledger = read_json(directory / "ledger.json")
    ledger["entries"] = []
    write_json(directory / "ledger.json", ledger)
    with pytest.raises(GenerationError, match="reservation_claim_ledger_mismatch"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert len(calls) == 16


def test_recorded_response_metadata_must_match_saved_raw(fixture_plan):
    directory = enable(fixture_plan)
    runner.run(fixture_plan["path"], permission=True, transports=transports([]))
    ledger = read_json(directory / "ledger.json")
    ledger["entries"][0]["response_id"] = "tampered"
    write_json(directory / "ledger.json", ledger)
    with pytest.raises(GenerationError, match="raw_response_metadata_mismatch"):
        runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])


def test_saved_voxels_must_equal_strict_raw_reexecution(fixture_plan):
    directory = enable(fixture_plan)
    runner.run(fixture_plan["path"], permission=True, transports=transports([]))
    ledger = read_json(directory / "ledger.json")
    descriptor = ledger["entries"][0]["artifacts"]["voxels.json"]
    artifact = directory / descriptor["path"]
    payload = runner.json_bytes(grid_to_payload(Grid({(1, 2, 3): "gold_block"})))
    artifact.write_bytes(payload)
    descriptor.update(sha256=runner.digest(payload), bytes=len(payload))
    write_json(directory / "ledger.json", ledger)
    with pytest.raises(GenerationError, match="successful_execution_artifact_mismatch"):
        runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])


def test_symlinked_artifact_rejected_even_if_target_is_inside_run(fixture_plan):
    directory = enable(fixture_plan)
    runner.run(fixture_plan["path"], permission=True, transports=transports([]))
    ledger = read_json(directory / "ledger.json")
    descriptor = ledger["entries"][0]["artifacts"]["raw.json"]
    artifact = directory / descriptor["path"]
    target = artifact.with_name("copied.json")
    artifact.rename(target)
    artifact.symlink_to(target)
    with pytest.raises(GenerationError, match="invalid_artifact_path"):
        runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])


def test_known_usage_reservation_violation_halts(fixture_plan):
    enable(fixture_plan)
    calls = []
    result = runner.run(fixture_plan["path"], permission=True, transports=transports(
        calls, lambda p, *_: response(p, usage={"input_tokens": 10, "output_tokens": 32769})))
    assert len(calls) == 1
    assert result["halt_reason"] == "budget_violation"
    assert result["used"]["new"]["tokens"] == 32779


def test_http_error_response_is_visible_but_usage_remains_unknown(fixture_plan):
    enable(fixture_plan)
    def failed(*_):
        raise providers.ProviderError("schema_unsupported", 400)
    result = runner.run(fixture_plan["path"], permission=True, transports=transports([], failed))
    assert result["halt_reason"] == "unknown_usage:schema_unsupported"
    summary = runner.evaluate_and_summarize(fixture_plan["path"], fixture_plan["grids"])
    assert summary["stage_counts"]["api_response_received"] == 1
    assert summary["stage_counts"]["completed_request_outcome"] == 0
    assert summary["quality_evaluated_inputs"] == 0


@pytest.mark.parametrize("name", ["approval.json", "enabled.json", "ledger.json"])
def test_symlinked_controls_rejected_before_send(fixture_plan, name):
    directory = enable(fixture_plan)
    path = directory / name
    target = directory.parent / ("foreign-" + name)
    path.rename(target)
    path.symlink_to(target)
    before = target.read_bytes()
    calls = []
    with pytest.raises(GenerationError, match="symlink_or_traversal"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert calls == [] and target.read_bytes() == before


@pytest.mark.parametrize("name", ["reservations", "attempts"])
def test_symlinked_output_directory_rejected_before_send(fixture_plan, name):
    directory = enable(fixture_plan)
    target = directory.parent / ("foreign-" + name)
    target.mkdir()
    (directory / name).symlink_to(target, target_is_directory=True)
    calls = []
    with pytest.raises(GenerationError, match="symlink_or_traversal"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert calls == [] and list(target.iterdir()) == []


def test_symlinked_immutable_claim_rejected_on_resume(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    path = next((directory / "reservations").glob("*.json"))
    target = directory.parent / "foreign-claim.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(GenerationError, match="symlink_or_traversal"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert len(calls) == 16


@pytest.mark.parametrize("kind", ["run", "parent", "authority"])
def test_symlinked_lock_is_not_opened_or_followed(fixture_plan, kind):
    directory = enable(fixture_plan)
    if kind == "run":
        path = directory / "run.lock"
    elif kind == "parent":
        path = fixture_plan["parent"] / "budget.lock"
    else:
        authority = Path(fixture_plan["plan"]["identity"]["authority_path"])
        path = authority.with_suffix(authority.suffix + ".lock")
    target = directory.parent / "foreign-lock"
    target.write_bytes(b"do not touch foreign lock")
    if path.exists():
        path.unlink()
    path.symlink_to(target)
    calls = []
    with pytest.raises(GenerationError, match="symlink_or_traversal"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls))
    assert calls == [] and target.read_bytes() == b"do not touch foreign lock"


def test_symlinked_attempt_destination_after_response_cannot_write_elsewhere(fixture_plan):
    directory = enable(fixture_plan)
    target = directory.parent / "foreign-attempt"
    target.mkdir()
    calls = []
    def replace_destination(provider, *_):
        attempt = read_json(directory / "ledger.json")["entries"][-1]
        (directory / "attempts").mkdir()
        (directory / "attempts" / attempt["attempt_id"]).symlink_to(target, target_is_directory=True)
        return response(provider)
    with pytest.raises(GenerationError, match="symlink_or_traversal"):
        runner.run(fixture_plan["path"], permission=True, transports=transports(calls, replace_destination))
    assert len(calls) == 1 and list(target.iterdir()) == []
    ledger = read_json(directory / "ledger.json")
    assert ledger["entries"][0]["status"] == "reserved"
    assert ledger["entries"][0]["usage"] is None
