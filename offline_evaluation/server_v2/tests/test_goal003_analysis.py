"""Offline saved-evidence regressions; all inputs and responses are synthetic.

Disabled preparations use the analysis-required new_implementation identity.
No generate call, provider credential, paid approval, or live transport is used.
The real run/artifact/metric integrity validators remain enabled.
"""
import json
import urllib.request

import pytest

from offline_evaluation.server_v2 import core, generation, providers
from offline_evaluation.server_v2.analysis_v4 import (
    attempted_failure, bounded_all_attempted_value, collect_planned_outcomes, summarize_v4,
)


REQUESTED = {"openai": "gpt-5-mini", "anthropic": "fixture-claude-explicit"}
SNAPSHOT = "gpt-5-mini-2025-08-07"
ACTIONS = {"operations": [{"op": "set", "x": 0, "y": 0, "z": 0, "block": "stone"}]}
GT = {f"fixture_only:scene_{i}": core.Grid({(0, 0, 0): "stone"}) for i in range(2)}


@pytest.fixture(autouse=True)
def prohibit_research_transport(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("saved-evidence analysis must not generate or use a transport")
    monkeypatch.setattr(generation, "generate", forbidden)
    monkeypatch.setattr(providers, "send_request", forbidden)
    monkeypatch.setattr(providers, "http_transport", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


def prepared_pair(tmp_path, *, approve_snapshot=True):
    records = [{"input_id": f"fixture-{scene}-{condition}", "scene_id": f"fixture_only:scene_{scene}",
                "condition": condition, "description_model": "openai" if condition in "OJ" else None,
                "status": "prepared", "dataset_version": "v4", "input_text": "One stone at the origin."}
               for scene in range(2) for condition in "OJPE"]
    directories = []
    for provider, model in REQUESTED.items():
        pricing = {"verified": True, "verified_on": "2026-10-13", "provider": provider, "model": model,
                   "context_window_tokens": 20_000, "input_usd_per_million": 1, "output_usd_per_million": 2,
                   "billing_mode": "standard_text_no_tools_no_cache_writes",
                   "rate_scope": "upper_bound_full_context_window"}
        if provider == "openai" and approve_snapshot:
            pricing["accepted_response_model_ids"] = [model, SNAPSHOT]
        config = generation.GenerationConfig(provider=provider, model=model, api_allowed=False,
                                              pricing=pricing, max_attempts_per_input=2)
        directory = tmp_path / provider
        generation.prepare_generation(records, config, directory, commit="a" * 40)
        directories.append(directory)
    return directories


def saved_attempt(directory, record_index, *, status="success", response_model=None, failure_kind=None):
    """Build an internally bound saved response; never invoke generate/HTTP."""
    manifest, config, records, prompt, state = generation._load_run(directory)
    record = records[record_index]
    identity = manifest["identity"]
    model = response_model or REQUESTED[config.provider]
    usage = {"input_tokens": 10, "output_tokens": 20}
    charged_input, charged_output, cost = providers.usage_upper_cost(config.provider, usage, config.pricing)
    attempt_id = f"{len(state['attempts']) + 1:032x}"
    payload = providers.build_payload(config.provider, config.model, prompt, record["input_text"], config.max_output_tokens)
    attempt = {"attempt_id": attempt_id, "input_id": record["input_id"], "scene_id": record["scene_id"],
               "condition": record["condition"], "identity": {
                   "run_identity_hash": generation.object_hash(identity), "input_hash": generation.object_hash(record),
                   "config_hash": identity["config_hash"], "prompt_hash": identity["prompt_hash"],
                   "commit": identity["commit"], "provider": config.provider, "model": config.model,
                   "request_payload_hash": generation.object_hash(payload)},
               "started_at_utc": "2026-10-13T00:00:00Z", "status": status, "failure_kind": failure_kind,
               "usage": usage, "response_id": f"fixture-response-{attempt_id}", "response_model": model,
               "charged_input_tokens": charged_input, "charged_output_tokens": charged_output,
               "charged_cost_usd": cost, "billing": "known_usage_conservative_cost", "artifacts": {}}
    text = json.dumps(ACTIONS) if status == "success" else "bad fixture JSON"
    raw = {"id": attempt["response_id"], "model": model, "usage": usage}
    if config.provider == "openai":
        raw.update(status="completed", output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}])
    else:
        raw.update(stop_reason="end_turn", content=[{"type": "text", "text": text}])
    generation._save_artifact(directory, attempt, "raw.json", generation.json_bytes(raw))
    if status == "success":
        generation._save_artifact(directory, attempt, "parsed.json", generation.json_bytes(ACTIONS))
        grid = core.execute(core.parse_actions(text))
        generation._save_artifact(directory, attempt, "voxels.json", generation.json_bytes(core.grid_to_payload(grid)))
    state["attempts"].append(attempt)
    generation._complete_attempt(directory, state, attempt)
    return attempt


def evaluated_pair(directories):
    for directory in directories:
        generation.evaluate_generation(directory, GT, alignment_mode="strict_translation")
    identities = {directory.name: generation.object_hash(generation.read_json(directory / "manifest.json")["identity"])
                  for directory in directories}
    return collect_planned_outcomes(directories, expected_identities=identities)


@pytest.mark.parametrize("provider,response_model", [
    ("openai", "gpt-5-mini"), ("openai", SNAPSHOT), ("anthropic", REQUESTED["anthropic"]),
])
@pytest.mark.parametrize("status", ["success", "failed"])
def test_frozen_exact_model_and_approved_snapshot_match_evaluation(tmp_path, provider, response_model, status):
    directories = prepared_pair(tmp_path)
    directory = next(path for path in directories if path.name == provider)
    saved_attempt(directory, 0, status=status, response_model=response_model,
                  failure_kind="parse_failure" if status == "failed" else None)
    rows = evaluated_pair(directories)
    row = next(row for row in rows if row["provider"] == provider and row["input_id"] == "fixture-0-O")
    assert len(rows) == 16 and row["completed_request_outcome"] is True
    assert row["status"] == status
    assert bounded_all_attempted_value(row) == (1.0 if status == "success" else 0.0)
    assert attempted_failure(row) is (status == "failed")
    assert sum(row["status"] == "NOT_RUN" for row in rows) == 15
    assert summarize_v4(rows, smoke=True)["research_requests_sent_by_analysis"] == 0


@pytest.mark.parametrize("approve_snapshot,response_model", [
    (False, SNAPSHOT), (True, "gpt-5-mini-2099-01-01"), (True, "gpt-5"),
])
def test_unapproved_snapshot_or_other_model_remains_unmeasurable(tmp_path, approve_snapshot, response_model):
    directories = prepared_pair(tmp_path, approve_snapshot=approve_snapshot)
    saved_attempt(directories[0], 0, status="failed", response_model=response_model,
                  failure_kind="response_identity_failure")
    rows = evaluated_pair(directories)
    row = rows[0]
    assert row["status"] == "failed" and row["completed_request_outcome"] is False
    assert not attempted_failure(row) and bounded_all_attempted_value(row) is None
    group = summarize_v4(rows, smoke=True)["groups"][0]
    assert group["all_attempted_rows"] == 0 and group["operational_or_unmeasurable_attempt_rows"] == 1
    assert group["failure_rate"] is None


def test_six_success_four_final_failure_six_not_run_collapse_fourteen_attempts(tmp_path):
    directories = prepared_pair(tmp_path)
    for index in range(8):
        if index <= 2:
            saved_attempt(directories[0], index, status="failed", response_model=SNAPSHOT, failure_kind="parse_failure")
        saved_attempt(directories[0], index, status="failed" if index in (1, 2) else "success",
                      response_model=SNAPSHOT, failure_kind="parse_failure" if index in (1, 2) else None)
    for index in (0, 0, 1):
        saved_attempt(directories[1], index, status="failed", failure_kind="parse_failure")
    rows = evaluated_pair(directories)
    assert [len(generation._load_run(path)[-1]["attempts"]) for path in directories] == [11, 3]
    assert len(rows) == 16 and sum(row["attempted"] for row in rows) == 10
    assert sum(row["status"] == "success" for row in rows) == 6
    assert sum(attempted_failure(row) for row in rows) == 4
    assert sum(row["status"] == "NOT_RUN" for row in rows) == 6
    assert all(row["metrics"] is None for row in rows if row["status"] != "success")
    summary = summarize_v4(rows, smoke=True)
    assert sum(group["planned_rows"] for group in summary["groups"]) == 16
    assert sum(group["all_attempted_rows"] for group in summary["groups"]) == 10
    assert sum(group["valid_rows"] for group in summary["groups"]) == 6
    assert sum(group["excluded_not_run_or_unmeasurable"] for group in summary["groups"]) == 6
    assert summary["families"] == [] and summary["smoke_inference_suppressed"] is True
    retried = next(row for row in rows if row["provider"] == "openai" and row["input_id"] == "fixture-0-O")
    assert retried["attempt_id"] == f"{2:032x}" and retried["status"] == "success"


@pytest.mark.parametrize("tamper,expected_error", [
    ("evaluation_input", "unknown or duplicate evaluation input identity"),
    ("evaluation_attempt", "final planned input/attempt identity"),
    ("evaluation_provider", "final planned input/attempt identity"),
    ("attempt_provider", "resume_attempt_identity_mismatch"),
    ("input_hash", "resume_identity_mismatch"),
    ("frozen_pricing", "resume_identity_mismatch"),
    ("raw_hash", "resume_output_integrity_mismatch"),
    ("metric_hash", "resume_output_integrity_mismatch"),
    ("metric_attempt", "metric artifact provenance mismatch"),
    ("metric_run", "metric artifact provenance mismatch"),
    ("metric_values", "metric artifact provenance mismatch"),
])
def test_approved_snapshot_does_not_bypass_real_identity_or_artifact_checks(tmp_path, tamper, expected_error):
    directories = prepared_pair(tmp_path)
    saved_attempt(directories[0], 0, response_model=SNAPSHOT)
    evaluated_pair(directories)
    directory = directories[0]
    evaluation_path = directory / "evaluation.json"
    evaluation = generation.read_json(evaluation_path)
    row = evaluation["rows"][0]
    if tamper.startswith("evaluation_"):
        field = {"evaluation_input": "input_id", "evaluation_attempt": "attempt_id", "evaluation_provider": "provider"}[tamper]
        row[field] = "wrong-fixture-identity"
        generation.write_json(evaluation_path, evaluation)
    elif tamper == "attempt_provider":
        state = generation.read_json(directory / "state.json")
        state["attempts"][0]["identity"]["provider"] = "anthropic"
        generation.write_json(directory / "state.json", state)
    elif tamper in {"input_hash", "frozen_pricing"}:
        path = directory / ("inputs.json" if tamper == "input_hash" else "config.json")
        data = generation.read_json(path)
        if tamper == "input_hash":
            data[0]["input_text"] = "Changed synthetic input"
        else:
            data["pricing"]["accepted_response_model_ids"] = [REQUESTED["openai"]]
        generation.write_json(path, data)
    elif tamper == "raw_hash":
        descriptor = generation.read_json(directory / "state.json")["attempts"][0]["artifacts"]["raw.json"]
        (directory / descriptor["path"]).write_bytes(b"changed synthetic raw")
    elif tamper == "metric_values":
        row["metrics"]["aligned"]["occupancy_iou"] = 0.25
        generation.write_json(evaluation_path, evaluation)
    else:
        descriptor = row["metrics_artifact"]
        path = directory / descriptor["path"]
        artifact = generation.read_json(path)
        artifact["attempt_id" if tamper != "metric_run" else "run_identity_hash"] = "wrong-fixture-identity"
        data = generation.json_bytes(artifact)
        path.write_bytes(data)
        if tamper != "metric_hash":
            # Even a consistent descriptor cannot authorize another metric identity.
            descriptor.update(sha256=generation.digest(data), bytes=len(data))
            generation.write_json(evaluation_path, evaluation)
    with pytest.raises((ValueError, generation.GenerationError), match=expected_error):
        collect_planned_outcomes(directories)
