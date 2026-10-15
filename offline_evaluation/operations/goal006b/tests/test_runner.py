"""Offline exercise of the separate, explicitly approved Goal006B ledger.

The shared plan fixture contains synthetic history. Live socket and HTTP paths
are blocked even when fixture transports are explicitly injected.
"""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from offline_evaluation.batch_v1.tests.test_runner import response, transports
from offline_evaluation.operations.goal006b import planning, runner
from offline_evaluation.resmoke_v2 import adapter
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.generation import GenerationError, object_hash, read_json, write_json


@pytest.fixture(autouse=True)
def forbid_live_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("live network or provider HTTP is forbidden in Goal006B fixtures")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(adapter, "http_transport", forbidden)
    monkeypatch.setattr(providers, "http_transport", forbidden)
    for name in providers.KEY_NAMES.values():
        monkeypatch.setenv(name, "goal006b_fixture_only_never_live")


def approval(plan):
    return {
        "manual_confirmed": True,
        "plan_identity_hash": plan["identity_hash"],
        "enabled_directory": plan["identity"]["enabled_directory"],
        "approved_at_utc": runner.utc_now(),
        "confirmation_channel": "/dev/tty",
        "scope": runner.APPROVAL_SCOPE,
        "acknowledges_unknown_hold": True,
    }


def verified_checkpoint(plan):
    identity = plan["identity"]
    reference = identity["checkpoint_reference"]
    return {
        "status": "GITHUB_VERIFIED",
        "verifiedledger_sha256": identity["source_snapshot"]["ledger_sha256"],
        "attempted_inputs": 781,
        "release_id": reference["release_id"],
        "assets": [{
            "name": "continuation-point-in-time.tar.gz",
            "asset_id": reference["archive_asset_id"],
            "sha256": reference["archive_sha256"],
        }],
    }


def saved(*_args, **_kwargs):
    return {"status": "LOCAL_VERIFIED"}


def enable(fixture):
    return runner.initialize_enabled(fixture["path"], approval=approval(fixture["plan"]))


def execute(fixture, calls, handler=None, **kwargs):
    return runner.run(
        fixture["path"], permission=True,
        transports=transports(calls, handler),
        checkpoint_verifier=kwargs.pop("checkpoint_verifier", verified_checkpoint),
        checkpoint_callback=kwargs.pop("checkpoint_callback", saved), **kwargs,
    )


def assert_originals_unchanged(fixture):
    assert all(Path(path).read_bytes() == content
               for path, content in fixture["source_before"].items())


def test_permission_and_disabled_approval_send_zero(fixture_plan):
    calls = []
    with pytest.raises(GenerationError):
        runner.run(fixture_plan["path"], transports=transports(calls),
                   checkpoint_verifier=verified_checkpoint, checkpoint_callback=saved)
    with pytest.raises(GenerationError):
        execute(fixture_plan, calls)
    assert calls == []
    assert not Path(fixture_plan["plan"]["identity"]["enabled_directory"]).exists()
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize("field,value", [
    ("scope", "goal006_fixed_evaluation_suffix800_only_cumulative_usd30"),
    ("acknowledges_unknown_hold", False),
    ("manual_confirmed", False),
    ("approved_at_utc", "2020-01-01T00:00:00Z"),
    ("confirmation_channel", "stdin"),
])
def test_wrong_scope_or_missing_acknowledgement_cannot_enable(fixture_plan, field, value):
    invalid = approval(fixture_plan["plan"])
    invalid[field] = value
    with pytest.raises(GenerationError):
        runner.initialize_enabled(fixture_plan["path"], approval=invalid)
    assert not Path(fixture_plan["plan"]["identity"]["enabled_directory"]).exists()
    assert_originals_unchanged(fixture_plan)


def test_existing_checkpoint_must_verify_before_first_reservation(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    try:
        result = execute(fixture_plan, calls, checkpoint_verifier=lambda _plan: {
            "status": "GITHUB_SAVE_FAILED", "reason": "fixture_digest_mismatch"})
    except GenerationError:
        pass
    else:
        assert result.get("operational_stop") or result.get("halt_reason")
    assert calls == []
    assert read_json(directory / "ledger.json")["entries"] == []
    assert_originals_unchanged(fixture_plan)


def test_exact20_original_payloads_one_manual_retry_and_no_repeat(fixture_plan):
    directory = enable(fixture_plan)
    calls, checkpoints = [], []
    original_identity = fixture_plan["source_plan"]["identity"]
    source_tail = original_identity["schedule"][780:]
    assert len(source_tail) == 20

    def inspect(provider, payload, count):
        expected = source_tail[count - 1]
        entry = read_json(directory / "ledger.json")["entries"][-1]
        assert entry["status"] == "reserved" and entry["usage"] is None
        assert entry["schedule_index"] == expected["schedule_index"] == 939 + count
        assert entry["input_id"] == expected["input_id"]
        assert entry["provider"] == expected["provider"] == provider
        assert entry["retry"] is (count == 1)
        assert entry["charged_output_tokens"] == 32768
        assert payload == adapter.build_payload(
            provider, expected["model"], original_identity["prompt"], expected["input_text"], 32768)
        assert entry["request_payload_hash"] == object_hash(payload)
        return response(provider)

    def checkpoint(plan, directory, ledger, reason):
        checkpoints.append((len(ledger["entries"]), reason))
        assert all(entry["status"] != "reserved" for entry in ledger["entries"])
        return saved()

    result = execute(fixture_plan, calls, inspect, checkpoint_callback=checkpoint)
    assert result["supplemental_attempted"] == 20 and result["supplemental_not_run"] == 0
    assert result["halt_reason"] is None
    assert result["combined_calls"] == 991
    assert len(calls) == 20
    entries = read_json(directory / "ledger.json")["entries"]
    assert [entry["schedule_index"] for entry in entries] == list(range(940, 960))
    assert sum(entry["retry"] for entry in entries) == 1
    assert entries[0]["source_attempt_id"] == fixture_plan["source_ledger"]["entries"][-1]["attempt_id"]
    assert all(entry["source_attempt_id"] is None for entry in entries[1:])
    assert result["combined_retries"] == 5 and result["automatic_retries"] == 0
    assert result["historical_unknown_usage_reservation_reconciled"] is False
    assert result["parent_budget_accounting"] == fixture_plan["plan"]["identity"]["parent"]["usage"]
    assert result["parent_budget_accounting"]["unknown_usage_attempts"] == 1
    assert result["combined_budget_cost_usd"] > result["parent_budget_accounting"]["cost_usd_upper_bound"]
    assert len({(entry["provider"], entry["input_id"]) for entry in entries}) == 20
    assert checkpoints and checkpoints[-1][0] == 20
    ledger_before = (directory / "ledger.json").read_bytes()
    execute(fixture_plan, calls)
    assert len(calls) == 20 and (directory / "ledger.json").read_bytes() == ledger_before
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize("kind,status_code", [("auth_failure", 401), ("timeout", None)])
def test_new_unknown_usage_halts_and_retains_reservation(fixture_plan, kind, status_code):
    directory = enable(fixture_plan)
    calls = []
    def fail(*_args):
        raise providers.ProviderError(kind, status_code)
    result = execute(fixture_plan, calls, fail)
    assert result["supplemental_attempted"] == 1 and result["supplemental_not_run"] == 19
    assert result["halt_reason"] == "unknown_usage:" + kind
    entry = read_json(directory / "ledger.json")["entries"][0]
    assert entry["usage"] is None and entry["billing"] == "worst_case_reservation"
    assert (entry["charged_input_tokens"], entry["charged_output_tokens"],
            entry["charged_cost_usd"]) == (400000, 32768, 0.165536)
    before = (directory / "ledger.json").read_bytes()
    execute(fixture_plan, calls)
    assert len(calls) == 1 and (directory / "ledger.json").read_bytes() == before
    assert_originals_unchanged(fixture_plan)


def test_interrupted_reserved_attempt_is_never_resent(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def interrupt(*_args):
        raise KeyboardInterrupt("fixture interruption after durable reservation")
    with pytest.raises(KeyboardInterrupt):
        execute(fixture_plan, calls, interrupt)
    ledger = read_json(directory / "ledger.json")
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["status"] == "reserved"
    assert ledger["entries"][0]["usage"] is None
    before = (directory / "ledger.json").read_bytes()
    try:
        execute(fixture_plan, calls)
    except GenerationError:
        pass
    assert len(calls) == 1
    resumed = read_json(directory / "ledger.json")
    assert resumed["entries"] == ledger["entries"]
    assert resumed["halt_reason"] == "unreconciled_supplemental_reservation"
    assert_originals_unchanged(fixture_plan)


def test_known_model_failure_is_recorded_once_then_next_input_runs(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def model_output(provider, _payload, count):
        if count == 1:
            return response(provider, status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return response(provider)
    result = execute(fixture_plan, calls, model_output)
    entries = read_json(directory / "ledger.json")["entries"]
    assert result["supplemental_attempted"] == 20 and len(calls) == 20
    assert entries[0]["status"] == "failed" and entries[0]["failure_kind"] == "truncation"
    assert entries[0]["usage"] is not None
    assert all(entry["status"] == "success" for entry in entries[1:])
    assert result["halt_reason"] is None
    assert_originals_unchanged(fixture_plan)


def test_checkpoint_failure_after_generation_never_resends_results(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    result = execute(fixture_plan, calls, checkpoint_callback=lambda *_a: {
        "status": "GITHUB_SAVE_FAILED", "reason": "fixture_download_digest_mismatch"})
    assert result["operational_stop"] == "checkpoint_save_failed"
    assert result["halt_reason"] is None
    assert result["checkpoint"]["status"] == "SAVE_FAILED"
    assert len(calls) == 20
    before = (directory / "ledger.json").read_bytes()
    failed_again = execute(fixture_plan, calls, checkpoint_callback=lambda *_a: {
        "status": "GITHUB_SAVE_FAILED", "reason": "fixture_download_digest_mismatch"})
    assert failed_again["operational_stop"] == "checkpoint_save_failed"
    assert failed_again["halt_reason"] is None
    assert failed_again["checkpoint"]["status"] == "SAVE_FAILED"
    recovered = execute(fixture_plan, calls)
    assert recovered["operational_stop"] is None
    assert len(calls) == 20 and (directory / "ledger.json").read_bytes() == before
    assert_originals_unchanged(fixture_plan)


def test_schema_failure_records_frozen_stage_and_continues_once(fixture_plan):
    directory = enable(fixture_plan)
    calls = []
    def model_output(provider, _payload, count):
        return response(provider, text='{"not_operations": []}') if count == 1 else response(provider)
    result = execute(fixture_plan, calls, model_output)
    entries = read_json(directory / "ledger.json")["entries"]
    first = entries[0]
    assert first["status"] == "failed" and first["failure_kind"] == "validation_failure"
    assert first["json_success"] is True and first["schema_success"] is False
    assert first["failure_stage"] == "schema" and first["executor_success"] is None
    assert first["usage"] is not None and set(first["artifacts"]) == {"raw.json"}
    assert all(entry["status"] == "success" for entry in entries[1:])
    assert result["halt_reason"] is None and len(calls) == 20
    execute(fixture_plan, calls)
    assert len(calls) == 20
    assert_originals_unchanged(fixture_plan)


def test_pricing_expires_after_first_send_refresh_resumes_only_remaining19(fixture_plan, monkeypatch):
    from offline_evaluation.resmoke_v2 import plan as semantic_plan
    from offline_evaluation.server_v2 import generation
    directory = enable(fixture_plan)
    calls, checkpoints = [], []
    document = deepcopy(fixture_plan["plan"]["identity"]["pricing_document"])
    tomorrow = (datetime.fromisoformat(document["verified_at_utc"].replace("Z", "+00:00"))
                + timedelta(days=1)).isoformat().replace("+00:00", "Z")
    controls = {name: (directory / name).read_bytes() for name in ("approval.json", "enabled.json")}
    plan_before = fixture_plan["path"].read_bytes()
    authority = Path(fixture_plan["plan"]["identity"]["authority_path"])
    authority_before = authority.read_bytes()

    class FixtureDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.fromisoformat(tomorrow.replace("Z", "+00:00"))
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    def advance_date(provider, _payload, count):
        assert count == 1
        monkeypatch.setattr(semantic_plan, "utc_now", lambda: tomorrow)
        monkeypatch.setattr(planning, "utc_now", lambda: tomorrow)
        monkeypatch.setattr(generation, "dt", SimpleNamespace(
            date=date, datetime=FixtureDateTime, timezone=timezone))
        return response(provider)

    def checkpoint(_plan, _directory, ledger, reason):
        checkpoints.append((len(ledger["entries"]), reason))
        return saved()

    stopped = execute(fixture_plan, calls, advance_date, checkpoint_callback=checkpoint)
    assert stopped["operational_stop"] == "pricing_freshness_stop"
    assert stopped["halt_reason"] is None
    assert stopped["supplemental_attempted"] == 1 and stopped["supplemental_not_run"] == 19
    assert checkpoints == [(1, "pricing_freshness_stop")]
    assert len(calls) == 1
    first_entry = deepcopy(read_json(directory / "ledger.json")["entries"][0])
    assert first_entry["schedule_index"] == 940 and first_entry["status"] == "success"
    with pytest.raises(GenerationError):
        execute(fixture_plan, calls)
    assert len(calls) == 1

    document["verified_at_utc"], document["verified_on"] = tomorrow, tomorrow[:10]
    for provider in adapter.MODELS:
        document[provider]["verified_on"] = tomorrow[:10]
    source = fixture_plan["path"].parent / "next-day-official-fixture.json"
    write_json(source, document)
    ledger_before_refresh = (directory / "ledger.json").read_bytes()
    planning.refresh_pricing(fixture_plan["path"], pricing_file=source)
    assert (directory / "ledger.json").read_bytes() == ledger_before_refresh
    assert fixture_plan["path"].read_bytes() == plan_before
    assert authority.read_bytes() == authority_before
    assert all((directory / name).read_bytes() == data for name, data in controls.items())

    resumed = execute(fixture_plan, calls, checkpoint_callback=checkpoint)
    entries = read_json(directory / "ledger.json")["entries"]
    assert resumed["supplemental_attempted"] == 20 and resumed["supplemental_not_run"] == 0
    assert resumed["halt_reason"] is None and resumed["operational_stop"] is None
    assert len(calls) == 20 and entries[0] == first_entry
    assert [entry["schedule_index"] for entry in entries[1:]] == list(range(941, 960))
    assert checkpoints == [(1, "pricing_freshness_stop"), (20, "supplement_finished")]
    assert_originals_unchanged(fixture_plan)


@pytest.mark.parametrize("corruption", ["payload", "duplicate", "cost"])
def test_mutated_child_ledger_cannot_resume(fixture_plan, corruption):
    directory = enable(fixture_plan)
    calls = []
    execute(fixture_plan, calls)
    path = directory / "ledger.json"
    ledger = read_json(path)
    if corruption == "payload":
        ledger["entries"][0]["request_payload_hash"] = "0" * 64
    elif corruption == "duplicate":
        ledger["entries"][1] = deepcopy(ledger["entries"][0])
    else:
        ledger["entries"][0]["charged_cost_usd"] = 0
    write_json(path, ledger)
    with pytest.raises(GenerationError):
        execute(fixture_plan, calls)
    assert len(calls) == 20
    assert_originals_unchanged(fixture_plan)


def test_copied_child_plan_cannot_start(fixture_plan):
    copied = fixture_plan["path"].with_name("copied-plan.json")
    copied.write_bytes(fixture_plan["path"].read_bytes())
    calls = []
    with pytest.raises(GenerationError):
        runner.run(copied, permission=True, transports=transports(calls),
                   checkpoint_verifier=verified_checkpoint, checkpoint_callback=saved)
    assert calls == []
    assert_originals_unchanged(fixture_plan)


def test_production_provenance_rejects_injected_transport(fixture_plan, monkeypatch):
    plan = deepcopy(fixture_plan["plan"])
    plan["identity"]["provenance"] = "new_implementation"
    plan["identity_hash"] = object_hash(plan["identity"])
    write_json(fixture_plan["path"], plan)
    monkeypatch.setattr(planning, "PROVENANCE", "new_implementation")
    calls = []
    with pytest.raises(GenerationError, match="injections_require_fixture_only"):
        execute(fixture_plan, calls)
    assert calls == []
    assert_originals_unchanged(fixture_plan)


def test_original_unknown_hold_counts_against_usd30_and_calls991(fixture_plan):
    identity = fixture_plan["plan"]["identity"]
    prior = identity["parent"]["usage"]
    held = identity["source_accounting"]["unknown_usage_held_reservation"]
    measured = identity["source_accounting"]["known_usage_conservative"]
    assert prior["calls"] == 971 and prior["unknown_usage_attempts"] == 1
    assert prior["tokens"] == measured["tokens"] + held["tokens"]
    assert prior["cost_usd_upper_bound"] == pytest.approx(
        measured["cost_usd_upper_bound"] + held["cost_usd_upper_bound"])
    assert held["tokens"] == 432768 and held["cost_usd_upper_bound"] == 0.165536
    assert identity["budgets"]["max_total_calls"] == 991
    assert identity["budgets"]["max_total_retries"] == 5
    assert identity["budgets"]["max_cost_usd"] == 30
    assert identity["budgets"]["max_automatic_retries"] == 0
    empty = {"entries": [], "halt_reason": None}
    reservation = runner.base._reservation(identity, identity["schedule"][0])
    assert runner.base._can_reserve(identity, empty, reservation)
    exhausted = deepcopy(identity)
    exhausted["parent"]["usage"]["cost_usd_upper_bound"] = 30 - reservation[2] + 0.000001
    assert not runner.base._can_reserve(exhausted, empty, reservation)
    exhausted = deepcopy(identity)
    exhausted["parent"]["usage"]["calls"] = 991
    assert not runner.base._can_reserve(exhausted, empty, reservation)
    assert_originals_unchanged(fixture_plan)
