"""CPU diagnosis fixtures; no paid approval, provider query, or transport."""
import json
import urllib.request

import pytest

from offline_evaluation.server_v2 import core, generation, providers, shared_budget, smoke
from offline_evaluation.server_v2 import goal003_diagnosis as diagnosis
from offline_evaluation.server_v2.tests.test_goal003_analysis import (
    ACTIONS, GT, SNAPSHOT, evaluated_pair, prepared_pair, saved_attempt,
)


@pytest.fixture(autouse=True)
def forbid_research_transport(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU diagnosis must not generate, query availability, or use HTTP")
    for module, name in ((generation, "generate"), (providers, "send_request"),
                         (providers, "http_transport"), (smoke, "generate"),
                         (smoke, "run_smoke_campaign"), (smoke, "dry_run_smoke_campaign"),
                         (urllib.request, "urlopen")):
        monkeypatch.setattr(module, name, forbidden)


@pytest.mark.parametrize("text,expected", [
    ('{"operations": []}', {"strict_json": True, "action_validation": "success", "execution": "success"}),
    ('{"operations": [', {"strict_json": False, "action_validation": "NOT_RUN", "execution": "NOT_RUN"}),
    ('{"operations": [], "explanation": "extra"}', {"strict_json": True, "action_validation": "failed", "execution": "NOT_RUN"}),
    ('{"operations":[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]}',
     {"strict_json": True, "action_validation": "failed", "execution": "NOT_RUN"}),
    ('{"operations":[{"op":"set","x":100001,"y":0,"z":0,"block":"stone"}]}',
     {"strict_json": True, "action_validation": "failed", "execution": "NOT_RUN"}),
    ('{"operations":[{"op":"set","x":0,"y":0,"z":0,"block":"Stone"}]}',
     {"strict_json": True, "action_validation": "failed", "execution": "NOT_RUN"}),
])
def test_saved_text_distinguishes_json_schema_types_coordinates_and_materials(text, expected):
    result = diagnosis.inspect_text(text)
    assert {key: result[key] for key in expected} == expected


def test_duplicate_key_error_does_not_echo_untrusted_key():
    result = diagnosis.inspect_text('{"fixture-private-name": 1, "fixture-private-name": 2}')
    assert result["error"] == {"kind": "duplicate_json_key"}
    assert "fixture-private-name" not in json.dumps(result)


def test_only_unambiguous_outer_fence_is_removed_for_sensitivity():
    text = json.dumps(ACTIONS)
    result = diagnosis.fence_sensitivity("```json\n" + text + "\n```")
    assert result["strict_json"] is True and result["execution"] == "success"
    assert result["primary_status_changed"] is False and result["diagnostic_only"] is True
    for value in ("prose\n```json\n" + text + "\n```", "```json\n" + text,
                  "```json\n" + text + "\n```\nprose", "```json\n```\n{}\n```"):
        assert diagnosis.fence_sensitivity(value) is None
    duplicate = diagnosis.fence_sensitivity('```json\n{"operations": [], "operations": []}\n```')
    assert duplicate["strict_json"] is False and duplicate["error"]["kind"] == "duplicate_json_key"
    truncated = diagnosis.fence_sensitivity('```json\n{"operations": [\n```')
    assert truncated["strict_json"] is False


def _attempt_fixture(tmp_path, provider, text, *, status="failed", failure_kind="parse_failure", response_status="completed", **raw_changes):
    config = generation.GenerationConfig(provider=provider, model="fixture-model")
    directory = tmp_path / provider
    directory.mkdir()
    attempt = {"attempt_id": "a" * 32, "input_id": "fixture-input", "response_model": config.model,
               "response_id": "fixture-response", "usage": {"input_tokens": 10, "output_tokens": 20},
               "charged_input_tokens": 10, "charged_output_tokens": 20, "charged_cost_usd": .01,
               "status": status, "failure_kind": failure_kind, "artifacts": {}}
    raw = {"id": attempt["response_id"], "model": config.model, "usage": attempt["usage"]}
    if provider == "openai":
        raw.update(status=response_status, output=[{"type": "message", "content": [{"type": "output_text", "text": text}]}])
    else:
        raw.update(stop_reason="end_turn", content=[{"type": "text", "text": text}])
    raw.update(raw_changes)
    attempt["usage"] = raw["usage"]
    generation._save_artifact(directory, attempt, "raw.json", generation.json_bytes(raw))
    return directory, config, attempt, {"scene_id": "fixture-scene", "condition": "O"}


def test_truncation_requires_explicit_raw_reason_and_reports_token_breakdown(tmp_path):
    args = _attempt_fixture(tmp_path, "openai", '{"operations":[', failure_kind="truncation",
                            response_status="incomplete", incomplete_details={"reason": "max_output_tokens"},
                            usage={"input_tokens": 10, "output_tokens": 20,
                                   "output_tokens_details": {"reasoning_tokens": 3}})
    result = diagnosis.attempt_diagnosis(*args, 1, 1)
    assert result["main_cause"] == "explicit_output_token_limit"
    assert result["reasoning_tokens"] == 3 and result["non_reasoning_output_tokens"] == 17
    assert result["natural_response_complete"] is False and result["strict_json"] is False


def test_natural_anthropic_stop_is_not_action_success(tmp_path):
    args = _attempt_fixture(tmp_path, "anthropic", "```json\n" + json.dumps(ACTIONS) + "\n```")
    result = diagnosis.attempt_diagnosis(*args, 1, 1)
    assert result["natural_response_complete"] and result["status"] == "failed"
    assert result["main_cause"] == "markdown_fence" and result["strict_json"] is False
    assert result["fence_only_sensitivity"]["execution"] == "success"


@pytest.mark.parametrize("field", ["response_id", "response_model", "usage"])
def test_raw_metadata_identity_or_usage_disagreement_is_rejected(tmp_path, field):
    args = _attempt_fixture(tmp_path, "anthropic", "bad JSON")
    args[2][field] = {"input_tokens": 100, "output_tokens": 20} if field == "usage" else "wrong-fixture-value"
    with pytest.raises(ValueError, match="raw response identity/usage"):
        diagnosis.attempt_diagnosis(*args, 1, 1)


def _diagnostic_fixture(tmp_path, monkeypatch):
    """Mock ONLY paid campaign approval/ledger authority, using disabled fixtures.

    Real run identity, raw/attempt/metric artifacts, collection, parsing and metric
    recomputation remain active. The production paid evidence is checked without
    these stubs by the independent Goal003 diagnostic invocation.
    """
    campaign, logs = tmp_path / "fixture-campaign", tmp_path / "fixture-logs"
    campaign.mkdir()
    logs.mkdir()
    directories = prepared_pair(campaign / "runs")
    saved_attempt(directories[0], 0, response_model=SNAPSHOT)
    evaluated_pair(directories)
    runs = {directory.name: {"path": f"runs/{directory.name}",
                            "identity_hash": generation.object_hash(generation.read_json(directory / "manifest.json")["identity"])}
            for directory in directories}
    identity = {"campaign_id": "fixture-only-campaign", "runs": runs, "budgets": {"max_retries": 4}}
    manifest = {"identity": identity, "identity_hash": generation.object_hash(identity)}
    attempt = generation._load_run(directories[0])[-1]["attempts"][0]
    ledger = {"halt_reason": None, "entries": [{**attempt, "provider": "openai", "retry": False}]}
    used = shared_budget.totals(ledger)
    for name, value in (("campaign.json", manifest), ("approval.json", {"fixture_only": True}), ("ledger.json", ledger)):
        generation.write_json(campaign / name, value)
    for name in ("budget.lock", "run.lock"):
        (campaign / name).touch()
    for name in diagnosis.LOG_NAMES:
        (logs / name).write_text("{}\n")
    generation.write_json(logs / "generation.log", {"identity_hash": manifest["identity_hash"], "used": used,
                                                   "providers": {"openai": {}, "anthropic": {}}})
    monkeypatch.setattr(shared_budget, "load_campaign", lambda directory: (manifest, ledger))
    monkeypatch.setattr(shared_budget.SharedBudget, "_validate_ledger", lambda self: None)
    return campaign, logs, directories


def test_complete_cpu_diagnostic_preserves_sources_and_emits_all_planned_rows(tmp_path, monkeypatch):
    campaign, logs, _ = _diagnostic_fixture(tmp_path, monkeypatch)
    before = diagnosis.source_inventory(campaign, logs)
    result = diagnosis.diagnose(campaign, logs, tmp_path / "new-diagnostic", gt_by_scene=GT)
    assert len(result["planned_outcomes"]) == 16
    assert sum(r["status"] == "NOT_RUN" for r in result["planned_outcomes"]) == 15
    assert len(result["metric_verification"]) == 1 and result["metric_verification"][0]["recomputed_exact_match"]
    assert result["summary"]["families"] == []
    assert result["ledger_reconciliation"]["research_api_additional_sends"] == 0
    assert result["ledger_reconciliation"]["experiment_complete"] is False
    assert result["availability_history"]["openai"]["identified_approved_model_response_attempts"] == 1
    assert result["availability_history"]["anthropic"]["current_or_future_availability"] == "NOT_QUERIED"
    assert diagnosis.source_inventory(campaign, logs) == before == result["original_hashes"]["after"]
    assert len(list((tmp_path / "new-diagnostic").glob("*.json"))) == 8
    assert len(list((tmp_path / "new-diagnostic").glob("*.csv"))) == 2


@pytest.mark.parametrize("tamper", ["extra_file", "extra_directory", "symlink", "missing_control"])
def test_inventory_rejects_unapproved_paths_or_missing_control_files(tmp_path, monkeypatch, tamper):
    campaign, logs, _ = _diagnostic_fixture(tmp_path, monkeypatch)
    if tamper == "extra_file":
        (campaign / "unapproved.txt").write_text("unapproved fixture file")
    elif tamper == "extra_directory":
        (campaign / "unapproved").mkdir()
    elif tamper == "symlink":
        (logs / "generation.log").unlink()
        (logs / "generation.log").symlink_to(logs / "summary.log")
    else:
        (campaign / "budget.lock").unlink()
    with pytest.raises(ValueError):
        diagnosis.source_inventory(campaign, logs)


@pytest.mark.parametrize("target", ["campaign_child", "logs_child", "existing", "ancestor", "symlink_ancestor", "parent_segments"])
def test_diagnostic_destination_cannot_overwrite_or_overlap_sources(tmp_path, monkeypatch, target):
    campaign, logs, _ = _diagnostic_fixture(tmp_path, monkeypatch)
    destinations = {"campaign_child": campaign / "diagnostic", "logs_child": logs / "diagnostic",
                    "existing": tmp_path / "existing", "ancestor": tmp_path}
    (tmp_path / "existing").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "existing", target_is_directory=True)
    destinations["symlink_ancestor"] = tmp_path / "link" / "diagnostic"
    destinations["parent_segments"] = tmp_path / "existing" / ".." / campaign.name / "diagnostic"
    with pytest.raises((ValueError, FileExistsError)):
        diagnosis.diagnose(campaign, logs, destinations[target], gt_by_scene=GT)


def test_metric_recompute_rejects_wrong_gt_without_rewriting_originals(tmp_path, monkeypatch):
    campaign, logs, _ = _diagnostic_fixture(tmp_path, monkeypatch)
    before = diagnosis.source_inventory(campaign, logs)
    with pytest.raises(ValueError, match="metric artifact/GT identity mismatch"):
        diagnosis.build_diagnosis(campaign, logs, gt_by_scene={key: core.Grid({}) for key in GT})
    assert diagnosis.source_inventory(campaign, logs) == before


def test_active_campaign_lock_blocks_diagnosis(tmp_path, monkeypatch):
    campaign, logs, _ = _diagnostic_fixture(tmp_path, monkeypatch)
    with diagnosis._existing_lock(campaign / "budget.lock"):
        with pytest.raises(ValueError, match="active campaign"):
            diagnosis.build_diagnosis(campaign, logs, gt_by_scene=GT)
