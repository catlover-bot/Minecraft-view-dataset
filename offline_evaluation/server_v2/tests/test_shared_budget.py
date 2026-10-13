"""fixture_only durable budget tests; no provider requests."""
import json
import uuid

import pytest

from offline_evaluation.server_v2.generation import (GenerationError, _complete_attempt, object_hash, read_json, utc_now, write_json)
from offline_evaluation.server_v2.providers import build_payload
from offline_evaluation.server_v2.shared_budget import BudgetStop, SharedBudget, campaign_summary
from offline_evaluation.server_v2.tests.test_smoke import fixture_enabled


def fixture_scope(enabled, provider):
    manifest = read_json(enabled / f"runs/{provider}/manifest.json")
    campaign = read_json(enabled / "campaign.json")
    return manifest["identity"], campaign["identity"]["campaign_id"]


def fixture_reserve(budget, provider, input_id, *, input_tokens=20_000, output_tokens=16_384, cost=0.1):
    run = budget.manifest["identity"]["runs"][provider]
    directory = budget.directory / run["path"]
    config = read_json(directory / "config.json")
    record = next(record for record in read_json(directory / "inputs.json") if record["input_id"] == input_id)
    payload = build_payload(provider, config["model"], (directory / "prompt.txt").read_text(), record["input_text"], config["max_output_tokens"])
    return budget.reserve(run_identity_hash=run["identity_hash"], provider=provider, input_id=input_id,
                          attempt_id=uuid.uuid4().hex, input_tokens=input_tokens, output_tokens=output_tokens,
                          cost_usd=cost, request_payload_hash=object_hash(payload))


def fixture_settle(budget, entry, *, known=True, cost=0.001):
    directory = budget.directory / f"runs/{entry['provider']}"
    identity = read_json(directory / "manifest.json")["identity"]
    record = next(record for record in read_json(directory / "inputs.json") if record["input_id"] == entry["input_id"])
    attempt = {"attempt_id": entry["attempt_id"], "input_id": entry["input_id"],
               "identity": {"run_identity_hash": entry["run_identity_hash"], "input_hash": object_hash(record),
                            "config_hash": identity["config_hash"], "prompt_hash": identity["prompt_hash"],
                            "commit": identity["commit"], "provider": entry["provider"], "model": identity["model"],
                            "request_payload_hash": entry["request_payload_hash"]},
               "scene_id": record["scene_id"], "condition": record["condition"], "started_at_utc": utc_now(),
               "status": "failed", "failure_kind": "parse_failure" if known else "timeout", "artifacts": {},
               "usage": {"input_tokens": 1, "output_tokens": 1} if known else None,
               "charged_input_tokens": 1 if known else entry["charged_input_tokens"],
               "charged_output_tokens": 1 if known else entry["charged_output_tokens"],
               "charged_cost_usd": cost if known else entry["charged_cost_usd"]}
    state = read_json(directory / "state.json")
    state["attempts"].append(attempt)
    if not known:
        state["halt_reason"] = "unknown_usage"
    _complete_attempt(directory, state, attempt)
    budget.settle(attempt)


def test_global20calls4retries_perinput2_across_restart(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    campaign = read_json(enabled / "campaign.json")
    target_ids = {provider: [target["input_id"] for target in campaign["identity"]["targets"] if target["provider"] == provider]
                  for provider in ("openai", "anthropic")}
    for provider in ("openai", "anthropic"):
        identity, campaign_id = fixture_scope(enabled, provider)
        with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
            for input_id in target_ids[provider]:
                fixture_settle(budget, fixture_reserve(budget, provider, input_id))
    identity, campaign_id = fixture_scope(enabled, "openai")
    for input_id in target_ids["openai"][:4]:
        with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
            fixture_settle(budget, fixture_reserve(budget, "openai", input_id))
    result = campaign_summary(enabled)
    assert result["used"]["calls"] == 20 and result["used"]["retries"] == 4
    with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
        with pytest.raises(BudgetStop, match="input_attempt_limit"):
            fixture_reserve(budget, "openai", target_ids["openai"][0])
        with pytest.raises(BudgetStop, match="call_limit"):
            fixture_reserve(budget, "openai", target_ids["openai"][4])
    assert campaign_summary(enabled)["used"]["calls"] == 20


def test_unknown_usage_retains_full_reservation_and_globally_stops(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    identity, campaign_id = fixture_scope(enabled, "openai")
    input_id = identity["inputs"][0]["input_id"]
    with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
        fixture_settle(budget, fixture_reserve(budget, "openai", input_id, cost=0.5), known=False)
    result = campaign_summary(enabled)
    assert result["used"]["tokens"] == 36_384 and result["used"]["cost_usd_upper_bound"] == 0.5
    identity, campaign_id = fixture_scope(enabled, "anthropic")
    with pytest.raises(BudgetStop, match="unknown_usage"):
        with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id):
            pytest.fail("globally halted ledger reused")


def test_crash_before_local_buffer_write_retains_reservation_and_stops_other_provider(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    identity, campaign_id = fixture_scope(enabled, "openai")
    with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
        fixture_reserve(budget, "openai", identity["inputs"][0]["input_id"])
    identity, campaign_id = fixture_scope(enabled, "anthropic")
    with pytest.raises(BudgetStop, match="unreconciled_prior_reservation"):
        with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id):
            pytest.fail("crashed request retried")
    assert campaign_summary(enabled)["used"]["calls"] == 1


def test_shared_cost_and_token_limits_fail_before_reservation(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    identity, campaign_id = fixture_scope(enabled, "openai")
    with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
        with pytest.raises(BudgetStop, match="cost_limit"):
            fixture_reserve(budget, "openai", identity["inputs"][0]["input_id"], cost=5.1)
        with pytest.raises(BudgetStop, match="token_limit"):
            fixture_reserve(budget, "openai", identity["inputs"][0]["input_id"], input_tokens=10_000_000)
    assert campaign_summary(enabled)["used"]["calls"] == 0


def test_shared_lock_blocks_concurrent_provider_requests(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    first, campaign_id = fixture_scope(enabled, "openai")
    second, _ = fixture_scope(enabled, "anthropic")
    with SharedBudget(enabled).locked(run_identity=first, campaign_id=campaign_id):
        with pytest.raises(GenerationError, match="locked"):
            with SharedBudget(enabled).locked(run_identity=second, campaign_id=campaign_id):
                pytest.fail("concurrent request enabled")


def test_arbitrary_run_cannot_join_approved_campaign(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    identity, campaign_id = fixture_scope(enabled, "openai")
    identity["commit"] = "b" * 40
    with pytest.raises(GenerationError, match="unapproved_run_identity"):
        with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id):
            pytest.fail("unapproved run accepted")


def test_other_provider_completed_output_integrity_checked_globally(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    identity, campaign_id = fixture_scope(enabled, "openai")
    with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
        entry = fixture_reserve(budget, "openai", identity["inputs"][0]["input_id"])
        fixture_settle(budget, entry)
    path = enabled / "runs/openai" / f"attempts/{entry['attempt_id']}/attempt.json"
    path.write_text("{}")
    second, _ = fixture_scope(enabled, "anthropic")
    with pytest.raises(GenerationError, match="integrity_mismatch"):
        with SharedBudget(enabled).locked(run_identity=second, campaign_id=campaign_id):
            pytest.fail("corrupted first-provider output overlooked")


def test_unknown_reservation_cannot_report_zero_after_single_file_corruption(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    identity, campaign_id = fixture_scope(enabled, "openai")
    with SharedBudget(enabled).locked(run_identity=identity, campaign_id=campaign_id) as budget:
        fixture_reserve(budget, "openai", identity["inputs"][0]["input_id"])
    ledger = read_json(enabled / "ledger.json")
    for name in ("charged_input_tokens", "charged_output_tokens", "charged_cost_usd"):
        ledger["entries"][0][name] = 0
    write_json(enabled / "ledger.json", ledger)
    with pytest.raises(GenerationError, match="below_verified_ceiling"):
        campaign_summary(enabled)
