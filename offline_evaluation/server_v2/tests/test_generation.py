"""fixture_only HTTP unit tests; no live provider calls or API-success claims."""

import datetime as dt
import json
from dataclasses import replace

import pytest

from offline_evaluation.server_v2 import core, providers
from offline_evaluation.server_v2 import generation
from offline_evaluation.server_v2.generation import (
    GenerationConfig, GenerationError, dry_run_generation, evaluate_generation,
    generate, prepare_generation, read_json, run_lock, write_json,
)


COMMIT = "a" * 40
ACTIONS = {"operations": [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"}]}


def fixture_config(provider="anthropic", **changes):
    """Rates/model/context here are synthetic fixture values, never live pricing."""
    host = "https://platform.claude.com" if provider == "anthropic" else "https://developers.openai.com"
    config = GenerationConfig(
        provider=provider, model="fixture-model-explicit", api_allowed=True,
        max_calls=6, max_tokens=100_000, max_cost_usd=100, max_output_tokens=128,
        provenance="fixture_only", pricing={
            "verified": True, "provider": provider, "model": "fixture-model-explicit",
            "verified_on": dt.datetime.now(dt.timezone.utc).date().isoformat(),
            "billing_mode": "standard_text_no_tools_no_cache_writes",
            "rate_scope": "upper_bound_full_context_window",
            "context_window_tokens": 2048, "input_usd_per_million": 1,
            "output_usd_per_million": 2, "pricing_source_url": host + "/fixture-pricing",
            "context_source_url": host + "/fixture-context",
        },
    )
    return replace(config, **changes)


def fixture_records(count=1):
    return [{"scene_id": f"fixture_only:building_{i}", "condition": "original",
             "input_text": "One stone at the origin.", "description_model": "fixture_only"}
            for i in range(count)]


def fixture_raw(provider="anthropic", text=None, usage=None, **changes):
    text = json.dumps(ACTIONS) if text is None else text
    usage = {"input_tokens": 10, "output_tokens": 20} if usage is None else usage
    data = {"id": "fixture-response-id", "model": "fixture-model-explicit", "usage": usage}
    if provider == "anthropic":
        data.update(content=[{"type": "text", "text": text}], stop_reason="end_turn")
    else:
        data.update(output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}], status="completed")
    data.update(changes)
    return json.dumps(data).encode()


@pytest.fixture(autouse=True)
def fixture_keys_and_no_live_http(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture_only_key")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture_only_key")
    def no_live(*args, **kwargs):
        pytest.fail("fixture_only tests must never send live HTTP")
    monkeypatch.setattr(providers, "http_transport", no_live)


def fixture_run(tmp_path, config=None, count=1):
    directory = tmp_path / "fixture_only_run"
    prepare_generation(fixture_records(count), config or fixture_config(), directory, commit=COMMIT)
    return directory


@pytest.mark.parametrize("permission", [False, True])
def test_api_disabled_sends_zero_requests(tmp_path, permission):
    calls = []
    directory = fixture_run(tmp_path, replace(fixture_config(), api_allowed=False))
    with pytest.raises(GenerationError, match="api_disabled"):
        generate(directory, permission=permission, transport=lambda *args: calls.append(args))
    assert calls == []
    assert dry_run_generation(directory)["attempt_count"] == 0
    assert read_json(directory / "state.json")["attempts"] == []


def test_explicit_permission_gate_sends_zero_requests(tmp_path):
    directory = fixture_run(tmp_path)
    with pytest.raises(GenerationError, match="permission"):
        generate(directory, transport=lambda *_: pytest.fail("request sent"))
    assert dry_run_generation(directory)["research_api_status"] == "NOT_RUN"


def test_low_level_provider_gate_sends_zero_requests():
    with pytest.raises(providers.ProviderError, match="api_disabled"):
        providers.send_request("anthropic", {}, transport=lambda *_: pytest.fail("request sent"))


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_fixture_only_generation_shared_executor_and_resume(tmp_path, provider):
    directory = fixture_run(tmp_path, fixture_config(provider))
    calls = []
    def transport(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == providers.ENDPOINTS[provider]
        assert payload["model"] == "fixture-model-explicit"
        assert payload["max_tokens" if provider == "anthropic" else "max_output_tokens"] == 128
        assert "tools" not in payload
        assert timeout == 60
        # Reservation must already exist before transport begins.
        attempt = read_json(directory / "state.json")["attempts"][0]
        assert attempt["status"] == "reserved"
        assert attempt["charged_input_tokens"] == 2048
        calls.append(payload)
        return fixture_raw(provider)
    result = generate(directory, permission=True, transport=transport)
    assert result["success_count"] == 1
    assert result["research_api_status"] == "FIXTURE_ONLY"
    assert result["budget"]["input_tokens"] == 10
    assert result["budget"]["output_tokens"] == 20
    assert generate(directory, permission=True, transport=transport)["attempt_count"] == 1
    assert len(calls) == 1
    gt = core.Grid({(0, 0, 0): "stone"})
    rows = evaluate_generation(directory, {"fixture_only:building_0": gt}, alignment_radius=0)
    assert rows[0]["metrics"]["absolute"]["occupancy_iou"] == 1
    attempt = read_json(directory / "state.json")["attempts"][0]
    assert {"raw.json", "parsed.json", "voxels.json"}.issubset(attempt["artifacts"])
    assert rows[0]["metrics_artifact"]["path"].startswith("evaluations/")
    assert attempt["response_id"] == "fixture-response-id"
    assert attempt["identity"]["commit"] == COMMIT
    assert attempt["started_at_utc"].endswith("Z")
    assert "fixture_only_key" not in (directory / "state.json").read_text()
    assert "Authorization" not in (directory / "state.json").read_text()
    assert dry_run_generation(directory)["success_count"] == 1


@pytest.mark.parametrize("change", [
    {"model": None}, {"max_calls": 0}, {"max_tokens": 0}, {"max_cost_usd": 0},
    {"pricing": None}, {"pricing": {"verified": False}},
])
def test_incomplete_generation_configuration_never_sends(tmp_path, change):
    directory = fixture_run(tmp_path, replace(fixture_config(), **change))
    with pytest.raises(GenerationError):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))


def test_stale_and_mismatched_model_pricing_never_sends(tmp_path):
    pricing = {**fixture_config().pricing, "model": "another-fixture-model"}
    directory = fixture_run(tmp_path, replace(fixture_config(), pricing=pricing))
    with pytest.raises(GenerationError, match="mismatch"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))
    pricing = {**fixture_config().pricing, "verified_on": "2001-01-01"}
    with pytest.raises(GenerationError, match="stale"):
        replace(fixture_config(), pricing=pricing).require_generation(True)


@pytest.mark.parametrize("change", [{"max_tokens": 2175}, {"max_cost_usd": 0.000001}])
def test_worst_case_reservation_prevents_budget_overrun(tmp_path, change):
    directory = fixture_run(tmp_path, replace(fixture_config(), **change))
    result = generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))
    assert result["generation_stop_reason"] == "budget_exhausted"
    assert result["attempt_count"] == 0


def test_known_usage_retry_counts_calls_across_resume(tmp_path):
    config = fixture_config(max_attempts_per_input=3, max_calls=1)
    directory = fixture_run(tmp_path, config)
    calls = []
    def transport(*_):
        calls.append(1)
        return fixture_raw(text="malformed fixture output")
    result = generate(directory, permission=True, transport=transport)
    assert result["attempt_count"] == 1
    assert result["generation_stop_reason"] == "budget_exhausted"
    assert read_json(directory / "state.json")["attempts"][0]["failure_kind"] == "parse_failure"
    result = generate(directory, permission=True, transport=transport)
    assert result["attempt_count"] == 1 and len(calls) == 1
    rows = evaluate_generation(directory, {})
    assert rows[0]["status"] == "failed" and rows[0]["metrics"] is None


def test_known_usage_retry_can_succeed_with_distinct_attempt_ids(tmp_path):
    directory = fixture_run(tmp_path, fixture_config(max_attempts_per_input=2))
    responses = iter([fixture_raw(text="bad fixture"), fixture_raw()])
    result = generate(directory, permission=True, transport=lambda *_: next(responses))
    assert result["attempt_count"] == 2 and result["success_count"] == 1
    attempts = read_json(directory / "state.json")["attempts"]
    assert attempts[0]["attempt_id"] != attempts[1]["attempt_id"]


@pytest.mark.parametrize("kind", ["timeout", "rate_limit", "transport_failure"])
def test_unknown_network_usage_retains_reservation_and_halts(tmp_path, kind):
    directory = fixture_run(tmp_path, fixture_config(max_attempts_per_input=3))
    def transport(*_):
        raise providers.ProviderError(kind, 429 if kind == "rate_limit" else None)
    result = generate(directory, permission=True, transport=transport)
    assert result["halt_reason"] == "unknown_usage"
    assert result["budget"]["input_tokens"] == 2048
    assert result["budget"]["output_tokens"] == 128
    assert result["budget"]["cost_usd_upper_bound"] > 0
    attempt = read_json(directory / "state.json")["attempts"][0]
    assert attempt["failure_kind"] == kind and attempt["usage"] is None
    with pytest.raises(GenerationError, match="run_halted"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("retry sent"))


@pytest.mark.parametrize("raw", [b"malformed fixture response", fixture_raw(usage={})])
def test_missing_usage_is_never_zero(tmp_path, raw):
    directory = fixture_run(tmp_path)
    result = generate(directory, permission=True, transport=lambda *_: raw)
    assert result["halt_reason"] == "unknown_usage"
    assert result["budget"]["unknown_usage_attempts"] == 1
    assert result["budget"]["cost_usd_upper_bound"] > 0


def test_crash_reservation_is_not_retried(tmp_path):
    directory = fixture_run(tmp_path)
    class FixtureCrash(BaseException):
        pass
    def transport(*_):
        raise FixtureCrash
    with pytest.raises(FixtureCrash):
        generate(directory, permission=True, transport=transport)
    with pytest.raises(GenerationError, match="unreconciled_prior_reservation"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("retry sent"))
    state = read_json(directory / "state.json")
    assert state["attempts"][0]["charged_input_tokens"] == 2048


def test_resume_checks_all_input_config_prompt_and_output_hashes(tmp_path):
    directory = fixture_run(tmp_path)
    generate(directory, permission=True, transport=lambda *_: fixture_raw())
    prepare_generation(fixture_records(), fixture_config(), directory, commit=COMMIT)
    with pytest.raises(GenerationError, match="identity_mismatch"):
        prepare_generation(fixture_records(), fixture_config(), directory, commit="b" * 40)
    attempt = read_json(directory / "state.json")["attempts"][0]
    path = directory / attempt["artifacts"]["voxels.json"]["path"]
    path.write_text("{}")
    with pytest.raises(GenerationError, match="integrity_mismatch"):
        dry_run_generation(directory)


@pytest.mark.parametrize("file", ["inputs.json", "config.json", "prompt.txt"])
def test_resume_input_and_config_tamper_rejected(tmp_path, file):
    directory = fixture_run(tmp_path)
    path = directory / file
    if file == "prompt.txt":
        path.write_text("changed fixture prompt")
    else:
        value = read_json(path)
        if file == "inputs.json":
            value[0]["input_text"] += " changed fixture"
        else:
            value["model"] = "changed-fixture-model"
        write_json(path, value)
    with pytest.raises(GenerationError, match="identity_mismatch"):
        dry_run_generation(directory)


def test_validation_and_execution_failures_are_distinct(tmp_path, monkeypatch):
    directory = fixture_run(tmp_path)
    result = generate(directory, permission=True, transport=lambda *_: fixture_raw(text='{"operations":[{"op":"shell"}]}'))
    assert result["success_count"] == 0
    assert read_json(directory / "state.json")["attempts"][0]["failure_kind"] == "validation_failure"
    directory = tmp_path / "fixture_only_execution"
    prepare_generation(fixture_records(), fixture_config(), directory, commit=COMMIT)
    def broken_execute(*args, **kwargs):
        raise RuntimeError("fixture executor failure")
    monkeypatch.setattr(core, "execute", broken_execute)
    generate(directory, permission=True, transport=lambda *_: fixture_raw())
    assert read_json(directory / "state.json")["attempts"][0]["failure_kind"] == "execution_failure"


def test_no_fallback_on_provider_model_mismatch(tmp_path):
    directory = fixture_run(tmp_path)
    generate(directory, permission=True, transport=lambda *_: fixture_raw(model="different-fixture-model"))
    attempt = read_json(directory / "state.json")["attempts"][0]
    assert attempt["failure_kind"] == "response_identity_failure"
    assert attempt["identity"]["model"] == "fixture-model-explicit"


def test_mock_transport_disallowed_for_research_provenance(tmp_path):
    directory = fixture_run(tmp_path, fixture_config(provenance="new_implementation"))
    with pytest.raises(GenerationError, match="fixture_only"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))


def test_input_length_no_silent_truncation(tmp_path):
    with pytest.raises(GenerationError, match="truncation"):
        fixture_run(tmp_path, fixture_config(max_input_bytes=1))


def test_concurrent_run_lock_rejected(tmp_path):
    directory = fixture_run(tmp_path)
    with run_lock(directory):
        with pytest.raises(GenerationError, match="run_locked"):
            dry_run_generation(directory)


def test_output_scope_rejected():
    with pytest.raises(ValueError, match="authorized"):
        prepare_generation(fixture_records(), fixture_config(), "/outside-authorized-research", commit=COMMIT)


def test_unknown_cache_write_usage_halts_conservatively(tmp_path):
    directory = fixture_run(tmp_path)
    generate(directory, permission=True, transport=lambda *_: fixture_raw(usage={
        "input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 1,
    }))
    assert read_json(directory / "state.json")["halt_reason"] == "unknown_usage"


def test_resume_request_payload_identity_mismatch_rejected(tmp_path):
    directory = fixture_run(tmp_path)
    generate(directory, permission=True, transport=lambda *_: fixture_raw())
    state = read_json(directory / "state.json")
    state["attempts"][0]["identity"]["request_payload_hash"] = "0" * 64
    write_json(directory / "state.json", state)
    with pytest.raises(GenerationError, match="attempt_identity_mismatch"):
        dry_run_generation(directory)


def test_normal_metadata_never_enters_provider_request(tmp_path):
    directory = tmp_path / "fixture_only_normal_metadata"
    records = fixture_records()
    records[0]["gt_occupied"] = 123456
    records[0]["stratum"] = 3
    prepare_generation(records, fixture_config(), directory, commit=COMMIT)
    def transport(request, _timeout):
        payload = json.loads(request.data)
        assert payload["messages"][0]["content"] == records[0]["input_text"]
        assert "gt_occupied" not in request.data.decode()
        assert "123456" not in request.data.decode()
        return fixture_raw()
    assert generate(directory, permission=True, transport=transport)["success_count"] == 1


def test_preparation_records_implementation_and_rejects_altered_hashes_before_transport(tmp_path):
    directory = fixture_run(tmp_path)
    manifest = read_json(directory / "manifest.json")
    assert manifest["identity"]["implementation_hashes"] == generation._implementation_hashes()
    assert {"generation.py", "providers.py", "core.py", "shared_budget.py", "smoke.py"}.issubset(manifest["identity"]["implementation_hashes"])
    manifest["identity"]["implementation_hashes"]["providers.py"] = "0" * 64
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(GenerationError, match="implementation_mismatch"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))
    assert read_json(directory / "state.json")["attempts"] == []


def test_resume_rejects_changed_executing_implementation_before_transport(tmp_path, monkeypatch):
    directory = fixture_run(tmp_path)
    hashes = generation._implementation_hashes()
    changed = {**hashes, "generation.py": "f" * 64}
    monkeypatch.setattr(generation, "_implementation_hashes", lambda: changed)
    with pytest.raises(GenerationError, match="implementation_mismatch"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))
    with pytest.raises(GenerationError, match="implementation_mismatch"):
        dry_run_generation(directory)
    assert read_json(directory / "state.json")["attempts"] == []


def test_strict_evaluation_records_mode_and_definitive_attempt_outcome(tmp_path):
    directory = fixture_run(tmp_path)
    generate(directory, permission=True, transport=lambda *_: fixture_raw())
    rows = evaluate_generation(directory, {"fixture_only:building_0": core.Grid({(20, -3, -17): "stone"})},
                               alignment_mode="strict_translation")
    assert rows[0]["provider"] == "anthropic"
    assert rows[0]["completed_request_outcome"] is True and rows[0]["attempted"] is True
    assert rows[0]["metrics"]["aligned"]["occupancy_iou"] == 1
    assert rows[0]["metrics"]["alignment"]["mode"] == "strict_translation"


def test_alignment_resource_limit_is_evaluation_null_not_model_failure(tmp_path, monkeypatch):
    from offline_evaluation.server_v2.alignment import AlignmentResourceLimit
    directory = fixture_run(tmp_path)
    generate(directory, permission=True, transport=lambda *_: fixture_raw())
    def limited(*args, **kwargs):
        raise AlignmentResourceLimit("fixture_only resource limit", {"fixture_only": True})
    monkeypatch.setattr(core, "evaluate_pair", limited)
    rows = evaluate_generation(directory, {"fixture_only:building_0": core.Grid({(0, 0, 0): "stone"})},
                               alignment_mode="strict_translation")
    assert rows[0]["status"] == "ALIGNMENT_RESOURCE_LIMIT" and rows[0]["metrics"] is None
    assert rows[0]["failure_kind"] is None and rows[0]["completed_request_outcome"] is True
    metadata = rows[0]["evaluation_failure"]["metadata"]
    assert metadata["fixture_only"] is True and metadata["search_complete"] is False
    assert metadata["resource_limit_reached"] is True


def fixture_openai_alias_config(*, accept_snapshot):
    config = fixture_config("openai")
    pricing = {**config.pricing, "model": "gpt-5-mini"}
    if accept_snapshot:
        pricing["accepted_response_model_ids"] = ["gpt-5-mini", "gpt-5-mini-2025-08-07"]
    return replace(config, model="gpt-5-mini", pricing=pricing)


def test_documented_explicit_snapshot_response_accepted_without_request_fallback(tmp_path):
    directory = fixture_run(tmp_path, fixture_openai_alias_config(accept_snapshot=True))
    sent = []
    def transport(request, _timeout):
        sent.append(json.loads(request.data)["model"])
        return fixture_raw("openai", model="gpt-5-mini-2025-08-07")
    assert generate(directory, permission=True, transport=transport)["success_count"] == 1
    assert sent == ["gpt-5-mini"]
    rows = evaluate_generation(directory, {"fixture_only:building_0": core.Grid({(0, 0, 0): "stone"})})
    assert rows[0]["completed_request_outcome"] is True


@pytest.mark.parametrize("accept_snapshot,response_model", [(False, "gpt-5-mini-2025-08-07"), (True, "gpt-5")])
def test_no_implicit_snapshot_acceptance_or_other_model_fallback(tmp_path, accept_snapshot, response_model):
    directory = fixture_run(tmp_path, fixture_openai_alias_config(accept_snapshot=accept_snapshot))
    sent = []
    def transport(request, _timeout):
        sent.append(json.loads(request.data)["model"])
        return fixture_raw("openai", model=response_model)
    assert generate(directory, permission=True, transport=transport)["success_count"] == 0
    attempt = read_json(directory / "state.json")["attempts"][0]
    assert attempt["failure_kind"] == "response_identity_failure" and sent == ["gpt-5-mini"]
    assert evaluate_generation(directory, {})[0]["completed_request_outcome"] is False


def test_undocumented_accepted_snapshot_rejected_before_transport(tmp_path):
    config = fixture_openai_alias_config(accept_snapshot=True)
    config = replace(config, pricing={**config.pricing, "accepted_response_model_ids": ["gpt-5-mini", "gpt-5-mini-2099-01-01"]})
    directory = fixture_run(tmp_path, config)
    with pytest.raises(GenerationError, match="undocumented"):
        generate(directory, permission=True, transport=lambda *_: pytest.fail("request sent"))
    assert read_json(directory / "state.json")["attempts"] == []
