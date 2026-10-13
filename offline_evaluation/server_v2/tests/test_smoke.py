"""fixture_only campaign tests: no live network or provider availability claim."""
import dataclasses
import datetime as dt
import json
import shutil

import pytest

from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.generation import (GenerationConfig, GenerationError, generate, object_hash,
                                                   prepare_generation, read_json, write_json)
from offline_evaluation.server_v2.smoke import (approval_template, approve_smoke_campaign,
                                              dry_run_smoke_campaign, prepare_smoke_campaign, run_smoke_campaign)


def fixture_records():
    rows = [{"scene_id": f"fixture_only:scene_{scene}", "condition": condition,
             "input_text": "One stone at the origin.", "source_sha256": "0" * 64,
             "description_model": "openai" if condition in "OJ" else None, "status": "prepared"}
            for scene in range(2) for condition in "OJPE"]
    return [{**row, "input_id": object_hash(row)} for row in rows]


def fixture_config(provider):
    host = "https://developers.openai.com" if provider == "openai" else "https://platform.claude.com"
    return GenerationConfig(provider=provider, model="fixture-model-explicit", provenance="fixture_only",
                            api_allowed=False, pricing={
                                "verified": True, "verified_on": dt.datetime.now(dt.timezone.utc).date().isoformat(),
                                "provider": provider, "model": "fixture-model-explicit", "context_window_tokens": 20_000,
                                "model_max_output_tokens": 65_536, "input_usd_per_million": 1, "output_usd_per_million": 2,
                                "billing_mode": "standard_text_no_tools_no_cache_writes", "rate_scope": "upper_bound_full_context_window",
                                "pricing_source_url": host + "/fixture-pricing", "context_source_url": host + "/fixture-context"})


def fixture_proposal(tmp_path):
    sources = {}
    for provider in ("openai", "anthropic"):
        sources[provider] = tmp_path / f"fixture_only_source_{provider}"
        prepare_generation(fixture_records(), fixture_config(provider), sources[provider], commit="a" * 40)
    directory = tmp_path / "fixture_only_proposal"
    prepare_smoke_campaign(sources, directory)
    return directory, sources


def fixture_enabled(tmp_path):
    proposal, sources = fixture_proposal(tmp_path)
    enabled = tmp_path / "fixture_only_enabled"
    approval = {**approval_template(proposal, enabled_directory=enabled), "manual_confirmed": True}
    approve_smoke_campaign(proposal, enabled, approval)
    return enabled, proposal, sources


def fixture_response(provider, *, text=None, usage=None, **changes):
    text = json.dumps({"operations": [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"}]}) if text is None else text
    data = {"id": "fixture-only-response", "model": "fixture-model-explicit",
            "usage": {"input_tokens": 10, "output_tokens": 20} if usage is None else usage}
    if provider == "openai":
        data.update(status="completed", output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}])
    else:
        data.update(stop_reason="end_turn", content=[{"type": "text", "text": text}])
    data.update(changes)
    return json.dumps(data).encode()


@pytest.fixture(autouse=True)
def no_live_network_and_fixture_keys(monkeypatch):
    for name in providers.KEY_NAMES.values():
        monkeypatch.setenv(name, "fixture_only_key")
    monkeypatch.setattr(providers, "http_transport", lambda *_: pytest.fail("live HTTP forbidden in fixture_only tests"))


def test_disabled_preparation_and_manual_approval_sends_zero_requests(tmp_path):
    proposal, sources = fixture_proposal(tmp_path)
    before = {provider: (directory / "config.json").read_bytes() for provider, directory in sources.items()}
    dry = dry_run_smoke_campaign(proposal)
    assert dry["api_allowed"] is False and dry["api_status"] == "NOT_RUN" and dry["target_count"] == 16
    assert dry["budgets"]["max_calls"] == 20 and dry["budgets"]["max_retries"] == 4
    assert dry["api_key_present"] == {"openai": True, "anthropic": True}
    assert "fixture_only_key" not in json.dumps(dry)
    with pytest.raises(GenerationError, match="campaign_api_disabled"):
        run_smoke_campaign(proposal, permission=True)
    with pytest.raises(GenerationError, match="manual"):
        approve_smoke_campaign(proposal, tmp_path / "denied", approval_template(proposal))
    enabled = tmp_path / "fixture_only_enabled"
    approve_smoke_campaign(proposal, enabled, {**approval_template(proposal, enabled_directory=enabled), "manual_confirmed": True})
    assert read_json(enabled / "ledger.json")["entries"] == []
    assert before == {provider: (directory / "config.json").read_bytes() for provider, directory in sources.items()}


@pytest.mark.parametrize("field", ["budgets", "models", "target_inputs", "prompt_hashes", "implementation_hashes", "enabled_config_hashes"])
def test_approval_is_bound_to_all_scope_identities(tmp_path, field):
    proposal, _ = fixture_proposal(tmp_path)
    approval = {**approval_template(proposal), "manual_confirmed": True, field: {}}
    with pytest.raises(GenerationError, match="matching manual"):
        approve_smoke_campaign(proposal, tmp_path / "denied", approval)
    assert not (proposal / "activation.json").exists()


def test_approval_not_inherited_by_new_proposal(tmp_path):
    first, _ = fixture_proposal(tmp_path / "first")
    second, _ = fixture_proposal(tmp_path / "second")
    with pytest.raises(GenerationError, match="matching manual"):
        approve_smoke_campaign(second, tmp_path / "denied", {**approval_template(first), "manual_confirmed": True})


def test_combined_provider_ledger_and_restart_never_regenerates_success(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    calls = {"openai": 0, "anthropic": 0}
    def transport_for(provider):
        def transport(request, _timeout):
            calls[provider] += 1
            assert len(read_json(enabled / "ledger.json")["entries"]) == sum(calls.values())
            assert json.loads(request.data)["max_output_tokens" if provider == "openai" else "max_tokens"] == 16_384
            return fixture_response(provider)
        return transport
    transports = {provider: transport_for(provider) for provider in calls}
    result = run_smoke_campaign(enabled, permission=True, transports=transports)
    assert result["used"]["calls"] == 16 and result["used"]["retries"] == 0
    assert result["api_status"] == "FIXTURE_ONLY" and result["full_experiment"] == "NOT_RUN"
    assert run_smoke_campaign(enabled, permission=True, transports=transports)["used"]["calls"] == 16
    assert calls == {"openai": 8, "anthropic": 8}


def test_shared_retry_count_survives_provider_and_restart(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    seen = 0
    def transport(_request, _timeout):
        nonlocal seen
        seen += 1
        return fixture_response("openai", text="bad fixture JSON" if seen % 2 else None)
    result = generate(enabled / "runs/openai", permission=True, campaign_dir=enabled, transport=transport)
    assert result["generation_stop_reason"] == "shared_retry_limit"
    ledger = read_json(enabled / "ledger.json")
    assert sum(entry["retry"] for entry in ledger["entries"]) == 4
    assert len(ledger["entries"]) == 9
    result = generate(enabled / "runs/anthropic", permission=True, campaign_dir=enabled,
                      transport=lambda *_: fixture_response("anthropic", text="bad fixture JSON"))
    assert result["generation_stop_reason"] == "shared_retry_limit"
    assert sum(entry["retry"] for entry in read_json(enabled / "ledger.json")["entries"]) == 4


@pytest.mark.parametrize("kind", ["timeout", "rate_limit", "auth_failure", "billing_failure", "model_unavailable", "context_failure"])
def test_uncertain_usage_globally_stops_both_providers(tmp_path, kind):
    enabled, _, _ = fixture_enabled(tmp_path)
    calls = []
    def failed(*_):
        calls.append("openai")
        raise providers.ProviderError(kind)
    result = run_smoke_campaign(enabled, permission=True, transports={
        "openai": failed, "anthropic": lambda *_: pytest.fail("second provider must be stopped")})
    assert result["used"]["calls"] == 1 and result["used"]["unknown_usage_attempts"] == 1
    assert result["used"]["cost_usd_upper_bound"] > 0
    assert result["halt_reason"] == "unknown_usage" and calls == ["openai"]
    with pytest.raises(GenerationError, match="halted"):
        run_smoke_campaign(enabled, permission=True, transports={"openai": lambda *_: pytest.fail("retry sent")})


def test_shared_campaign_required_and_no_second_activation(tmp_path):
    enabled, proposal, _ = fixture_enabled(tmp_path)
    with pytest.raises(GenerationError, match="campaign required"):
        generate(enabled / "runs/openai", permission=True, transport=lambda *_: pytest.fail("request sent"))
    with pytest.raises(GenerationError, match="already activated"):
        other = tmp_path / "other_enabled"
        approve_smoke_campaign(proposal, other, {**approval_template(proposal, enabled_directory=other), "manual_confirmed": True})
    copy = tmp_path / "copied_campaign"
    shutil.copytree(enabled, copy)
    with pytest.raises(GenerationError, match="ledger_directory_mismatch"):
        run_smoke_campaign(copy, permission=True, transports={"openai": lambda *_: pytest.fail("request sent")})
    run_copy = tmp_path / "copied_provider_run"
    shutil.copytree(enabled / "runs/openai", run_copy)
    with pytest.raises(GenerationError, match="unapproved_run_directory"):
        generate(run_copy, campaign_dir=enabled, permission=True, transport=lambda *_: pytest.fail("request sent"))


def test_missing_one_key_stops_before_either_request(tmp_path, monkeypatch):
    enabled, _, _ = fixture_enabled(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(GenerationError, match="missing_api_key"):
        run_smoke_campaign(enabled, permission=True, transports={"openai": lambda *_: pytest.fail("request sent")})
    assert read_json(enabled / "ledger.json")["entries"] == []


def test_manual_enabled_campaign_still_needs_invocation_permission(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    with pytest.raises(GenerationError, match="permission"):
        run_smoke_campaign(enabled, transports={"openai": lambda *_: pytest.fail("request sent")})


@pytest.mark.parametrize("status,body,expected", [
    (401, {}, "auth_failure"), (402, {}, "billing_failure"),
    (429, {"error": {"code": "insufficient_quota"}}, "billing_failure"),
    (429, {}, "rate_limit"), (404, {}, "model_unavailable"),
    (400, {"error": {"code": "context_length_exceeded"}}, "context_failure"),
])
def test_safe_http_error_classification(status, body, expected):
    assert providers.classify_http_error(status, json.dumps(body).encode()) == expected


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_truncation_classified_separately(provider):
    changes = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}} if provider == "openai" else {"stop_reason": "max_tokens"}
    response = providers.parse_response(provider, fixture_response(provider, **changes))
    assert not response.complete and response.failure_kind == "truncation"


@pytest.mark.parametrize("rollback", ["empty", "older"])
def test_partial_restore_of_global_ledger_rejected_before_new_provider_request(tmp_path, rollback):
    enabled, _, _ = fixture_enabled(tmp_path)
    initial = read_json(enabled / "ledger.json")
    generate(enabled / "runs/openai", permission=True, campaign_dir=enabled,
             transport=lambda *_: fixture_response("openai"))
    changed = initial if rollback == "empty" else read_json(enabled / "ledger.json")
    if rollback == "older":
        changed["entries"] = changed["entries"][:1]
    write_json(enabled / "ledger.json", changed)
    with pytest.raises(GenerationError, match="local_attempt_missing_from_shared_ledger"):
        generate(enabled / "runs/anthropic", permission=True, campaign_dir=enabled,
                 transport=lambda *_: pytest.fail("request sent after ledger rollback"))
    assert read_json(enabled / "runs/anthropic/state.json")["attempts"] == []


def test_generate_evaluate_summarize_resume_preserves_attempt_metadata(tmp_path):
    from offline_evaluation.server_v2.core import Grid
    from offline_evaluation.server_v2.generation import evaluate_generation
    from offline_evaluation.server_v2.shared_budget import campaign_summary
    enabled, _, _ = fixture_enabled(tmp_path)
    run_smoke_campaign(enabled, permission=True, transports={provider: (lambda *_args, p=provider: fixture_response(p))
                                                            for provider in ("openai", "anthropic")})
    before = {path: path.read_bytes() for path in enabled.glob("runs/*/attempts/*/attempt.json")}
    gt = {f"fixture_only:scene_{scene}": Grid({(0, 0, 0): "stone"}) for scene in range(2)}
    for provider in ("openai", "anthropic"):
        first = evaluate_generation(enabled / f"runs/{provider}", gt, alignment_mode="strict_translation")
        assert evaluate_generation(enabled / f"runs/{provider}", gt, alignment_mode="strict_translation") == first
    assert before == {path: path.read_bytes() for path in before}
    assert campaign_summary(enabled)["used"]["calls"] == 16
    result = run_smoke_campaign(enabled, permission=True, transports={provider: (lambda *_: pytest.fail("successful output regenerated"))
                                                                    for provider in ("openai", "anthropic")})
    assert result["used"]["calls"] == 16


def test_restored_activation_sentinel_cannot_reuse_approval_in_another_directory(tmp_path):
    enabled, proposal, _ = fixture_enabled(tmp_path)
    approved = read_json(enabled / "approval.json")
    (proposal / "activation.json").unlink()  # fixture_only simulation of a preactivation restore
    with pytest.raises(GenerationError, match="matching manual"):
        approve_smoke_campaign(proposal, tmp_path / "other_enabled", approved)
    assert read_json(enabled / "ledger.json")["entries"] == []


def test_known_critical_failure_halt_cannot_be_cleared(tmp_path):
    enabled, _, _ = fixture_enabled(tmp_path)
    generate(enabled / "runs/openai", permission=True, campaign_dir=enabled,
             transport=lambda *_: fixture_response("openai", model="wrong-fixture-model"))
    ledger = read_json(enabled / "ledger.json")
    assert ledger["halt_reason"] == "response_identity_failure"
    ledger["halt_reason"] = None
    write_json(enabled / "ledger.json", ledger)
    with pytest.raises(GenerationError, match="critical_failure_halt_missing"):
        generate(enabled / "runs/anthropic", permission=True, campaign_dir=enabled,
                 transport=lambda *_: pytest.fail("critical failure stop cleared"))
