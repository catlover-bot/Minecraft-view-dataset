"""CPU-only forensic diagnosis of an existing, identity-bound smoke campaign.

No generation, availability query, reservation, or dry-run writer is called.
Evidence is read without replacement. The existing collector may open existing
provider run.lock control files; those are inventoried separately from evidence.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re

from . import analysis_v4, generation, providers, shared_budget
from .checkpoint import scan_file
from .core import Limits, ValidationError, evaluate_pair, execute, grid_to_payload, parse_actions, payload_to_grid, strict_json
from .io_utils import sha256, writable

LOG_NAMES = frozenset({"approval.json", "approval_template.json", "activation_result.json", "generation.log",
                       "evaluate_openai.log", "evaluate_anthropic.log", "summary.log", "final_status.log"})
RUN_NAMES = frozenset({"manifest.json", "config.json", "state.json", "inputs.json", "prompt.txt", "evaluation.json"})


def _safe_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink source or destination is forbidden")
    return path.resolve(strict=False)


def _relative_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must remain within its evidence directory")
    path = _safe_path(root / relative)
    if not path.is_file():
        raise ValueError("required evidence/control file is missing or not regular")
    return path


def source_inventory(campaign_dir, logs_dir):
    """Allowlist every paid artifact and original log; reject untracked files."""
    campaign, logs = _safe_path(campaign_dir), _safe_path(logs_dir)
    if campaign == logs or campaign.is_relative_to(logs) or logs.is_relative_to(campaign):
        raise ValueError("campaign and log evidence directories must be separate")
    expected = {"campaign.json", "approval.json", "ledger.json"}
    controls = {"budget.lock", "run.lock"}
    manifest = generation.read_json(_relative_file(campaign, "campaign.json"))
    if set(manifest["identity"]["runs"]) != set(analysis_v4.PROVIDERS):
        raise ValueError("exactly two approved provider runs required")
    for provider, run in manifest["identity"]["runs"].items():
        if run["path"] != f"runs/{provider}":
            raise ValueError("unexpected provider evidence directory")
        prefix = Path(run["path"])
        directory = campaign / prefix
        expected.update((prefix / name).as_posix() for name in RUN_NAMES)
        controls.add((prefix / "run.lock").as_posix())
        state = generation.read_json(_relative_file(directory, "state.json"))
        for attempt in state["attempts"]:
            attempt_id = attempt["attempt_id"]
            if not re.fullmatch(r"[0-9a-f]{32}", attempt_id):
                raise ValueError("invalid attempt identifier in evidence inventory")
            for name, descriptor in attempt.get("artifacts", {}).items():
                if name not in {"raw.json", "parsed.json", "voxels.json"} or descriptor["path"] != f"attempts/{attempt_id}/{name}":
                    raise ValueError("unapproved attempt evidence path")
                expected.add((prefix / descriptor["path"]).as_posix())
            metadata = attempt.get("metadata_artifact")
            if metadata:
                if metadata["path"] != f"attempts/{attempt_id}/attempt.json":
                    raise ValueError("unapproved attempt metadata path")
                expected.add((prefix / metadata["path"]).as_posix())
        evaluation = generation.read_json(_relative_file(directory, "evaluation.json"))
        for row in evaluation["rows"]:
            descriptor = row.get("metrics_artifact")
            if descriptor:
                if not re.fullmatch(r"evaluations/[0-9a-f]{64}/metrics.json", descriptor["path"]):
                    raise ValueError("unapproved metric evidence path")
                expected.add((prefix / descriptor["path"]).as_posix())
    output = {"evidence": [], "controls": []}
    for label, root, allowlist, control_list in (("campaign", campaign, expected, controls),
                                                  ("logs", logs, LOG_NAMES, set())):
        actual = set()
        allowed_directories = {parent.as_posix() for name in allowlist | control_list
                               for parent in Path(name).parents if parent != Path(".")}
        for path in root.rglob("*"):
            _safe_path(path)
            if path.is_dir():
                if path.relative_to(root).as_posix() not in allowed_directories:
                    raise ValueError("unapproved directory within evidence root")
                continue
            if not path.is_file():
                raise ValueError("nonregular evidence path")
            actual.add(path.relative_to(root).as_posix())
        if actual != allowlist | control_list:
            raise ValueError("evidence file set differs from the explicit allowlist")
        for relative in sorted(actual):
            path = _relative_file(root, relative)
            scan_file(path)
            kind = "controls" if relative in control_list else "evidence"
            output[kind].append({"source": label, "path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return output


@contextmanager
def _existing_lock(path):
    path = _safe_path(path)
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("active campaign operation; diagnosis requires idle originals") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _safe_validation_error(exc):
    # Duplicate keys originate in untrusted output; never echo the key itself.
    message = str(exc)
    if message.startswith("duplicate JSON key:"):
        return {"kind": "duplicate_json_key"}
    result = {"kind": "validation_error", "detail": message}
    if isinstance(exc.__cause__, json.JSONDecodeError):
        cause = exc.__cause__
        result = {"kind": "invalid_json", "detail": cause.msg, "line": cause.lineno,
                  "column": cause.colno, "position": cause.pos}
    return result


def inspect_text(text, limits=None):
    """Parse, validate and execute saved text; return only structural details."""
    limits = limits or Limits()
    result = {"strict_json": False, "action_validation": "NOT_RUN", "execution": "NOT_RUN"}
    if text is None:
        return dict(result, error={"kind": "no_extracted_text"})
    try:
        strict_json(text, limits)
    except ValidationError as exc:
        return dict(result, error=_safe_validation_error(exc))
    result["strict_json"] = True
    try:
        actions = parse_actions(text, limits)
    except ValidationError as exc:
        return dict(result, action_validation="failed", error=_safe_validation_error(exc))
    result.update(action_validation="success", operation_count=len(actions))
    try:
        grid = execute(actions, limits)
    except ValidationError as exc:
        return dict(result, execution="failed", error=_safe_validation_error(exc))
    return dict(result, execution="success", voxel_count=len(grid.voxels))


def fence_sensitivity(text, limits=None):
    """Only remove one unambiguous complete outer Markdown code fence."""
    if text is None:
        return None
    match = re.fullmatch(r"\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*", text)
    if not match or "```" in match[1]:
        return None
    return {"transformation": "one_outer_markdown_fence_removed", "primary_status_changed": False,
            "diagnostic_only": True, **inspect_text(match[1], limits)}


def attempt_diagnosis(directory, config, attempt, record, attempt_number, ledger_index):
    raw_path = generation._verify_artifact(directory, attempt["artifacts"]["raw.json"])
    raw_bytes = raw_path.read_bytes()
    raw = json.loads(raw_bytes)
    response = providers.parse_response(config.provider, raw_bytes)
    if (response.response_id != attempt["response_id"] or response.model != attempt["response_model"]
            or response.usage != attempt["usage"]):
        raise ValueError("raw response identity/usage differs from immutable attempt")
    limits = Limits(**config.limits)
    inspection = inspect_text(response.text, limits)
    fence = fence_sensitivity(response.text, limits)
    if config.provider == "openai":
        text_blocks = [b for item in raw.get("output", []) if item.get("type") == "message"
                       for b in item.get("content", []) if b.get("type") == "output_text"]
        refusals = sum(b.get("type") == "refusal" for item in raw.get("output", [])
                       for b in item.get("content", []))
        stop, incomplete = raw.get("status"), (raw.get("incomplete_details") or {}).get("reason")
    else:
        text_blocks = [b for b in raw.get("content", []) if b.get("type") == "text"]
        refusals = sum(b.get("type") == "refusal" for b in raw.get("content", []))
        stop, incomplete = raw.get("stop_reason"), None
    reason_tokens = (response.usage or {}).get("output_tokens_details", {}).get("reasoning_tokens")
    output_tokens = (response.usage or {}).get("output_tokens")
    if attempt["status"] == "success":
        if inspection["execution"] != "success" or not response.complete:
            raise ValueError("saved successful attempt fails CPU action revalidation")
        actions = parse_actions(response.text, limits)
        if generation.read_json(generation._verify_artifact(directory, attempt["artifacts"]["parsed.json"])) != {"operations": actions}:
            raise ValueError("saved parsed actions differ from raw text")
        if generation.read_json(generation._verify_artifact(directory, attempt["artifacts"]["voxels.json"])) != grid_to_payload(execute(actions, limits)):
            raise ValueError("saved voxels differ from CPU execution of saved text")
        cause = "success"
    elif response.failure_kind == "truncation":
        cause = "explicit_output_token_limit"
    elif not response.complete:
        cause = "provider_response_incomplete"
    elif response.text is None:
        cause = "no_provider_output_text"
    elif not inspection["strict_json"]:
        cause = "markdown_fence" if fence and fence["execution"] == "success" else (
            "markdown_fence_and_" + fence.get("error", {}).get("kind", "action_failure") if fence else inspection["error"]["kind"])
    elif inspection["action_validation"] == "failed":
        cause = "action_schema_or_value_violation"
    elif inspection["execution"] == "failed":
        cause = "action_execution_failure"
    else:
        cause = attempt["failure_kind"] or "unclassified_recorded_failure"
    return {"provider": config.provider, "input_id": attempt["input_id"], "scene_id": record["scene_id"],
            "condition": record["condition"], "attempt_id": attempt["attempt_id"], "attempt_number": attempt_number,
            "ledger_index": ledger_index, "requested_model": config.model, "response_model": response.model,
            "response_id": response.response_id, "approved_response_model": response.model in providers.accepted_response_model_ids(config.provider, config.model, config.pricing),
            "status": attempt["status"], "failure_kind": attempt["failure_kind"], "stop_reason": stop,
            "incomplete_reason": incomplete, "natural_response_complete": response.complete,
            "input_tokens": attempt["charged_input_tokens"], "output_tokens": attempt["charged_output_tokens"],
            "reasoning_tokens": reason_tokens,
            "non_reasoning_output_tokens": output_tokens - reason_tokens if type(output_tokens) is int and type(reason_tokens) is int else None,
            "output_token_detail_interpretation": "non_reasoning is a reported-token subtraction, not a separately measured visible-text token count",
            "text_characters": len(response.text) if response.text is not None else 0,
            "text_bytes": len(response.text.encode()) if response.text is not None else 0,
            "extracted_text_blocks": len(text_blocks), "refusal_blocks": refusals,
            "raw_sha256": sha256(raw_path), "cost_usd_upper_bound": attempt["charged_cost_usd"],
            **inspection, "fence_only_sensitivity": fence, "main_cause": cause}


def _verify_metrics(directory, manifest, state, planned, gt_by_scene):
    result = []
    attempts = {a["attempt_id"]: a for a in state["attempts"]}
    evaluation = generation.read_json(directory / "evaluation.json")
    for row in planned:
        if row["status"] != "success":
            continue
        artifact = generation.read_json(generation._verify_artifact(directory, row["metrics_artifact"]))
        gt = gt_by_scene[row["scene_id"]]
        identity = {key: artifact[key] for key in ("alignment_radius", "alignment_mode", "gt_hash", "attempt_id", "run_identity_hash")}
        if (identity["gt_hash"] != generation.object_hash(grid_to_payload(gt))
                or identity["attempt_id"] != row["attempt_id"]
                or identity["run_identity_hash"] != generation.object_hash(manifest["identity"])
                or identity["alignment_radius"] != evaluation["alignment_radius"]
                or identity["alignment_mode"] != evaluation["alignment_mode"]
                or row["metrics_artifact"]["path"] != f"evaluations/{generation.object_hash(identity)}/metrics.json"):
            raise ValueError("metric artifact/GT identity mismatch")
        attempt = attempts[row["attempt_id"]]
        grid = payload_to_grid(generation.read_json(generation._verify_artifact(directory, attempt["artifacts"]["voxels.json"])))
        recomputed = evaluate_pair(gt, grid, alignment_radius=identity["alignment_radius"], alignment_mode=identity["alignment_mode"])
        if recomputed != artifact["metrics"] or artifact["metrics"] != row["metrics"]:
            raise ValueError("CPU recomputed metrics differ from saved metrics")
        result.append({"provider": row["provider"], "input_id": row["input_id"], "scene_id": row["scene_id"],
                       "condition": row["condition"], "attempt_id": row["attempt_id"],
                       "metrics_sha256": row["metrics_artifact"]["sha256"], **identity,
                       "recomputed_exact_match": True})
    return result


def build_diagnosis(campaign_dir, logs_dir, *, gt_by_scene):
    """Validate all originals and return diagnosis payloads without writing evidence."""
    campaign, logs = _safe_path(campaign_dir), _safe_path(logs_dir)
    before = source_inventory(campaign, logs)
    with ExitStack() as stack:
        stack.enter_context(_existing_lock(campaign / "run.lock"))
        stack.enter_context(_existing_lock(campaign / "budget.lock"))
        manifest, ledger = shared_budget.load_campaign(campaign)
        budget = shared_budget.SharedBudget(campaign)
        budget.manifest, budget.ledger = manifest, ledger
        budget._validate_ledger()  # Pure validation; no locked(), reserve(), settle(), or save().
        used = shared_budget.totals(ledger)
        runs = {p: campaign / run["path"] for p, run in manifest["identity"]["runs"].items()}
        expected = {p: run["identity_hash"] for p, run in manifest["identity"]["runs"].items()}
        planned = analysis_v4.collect_planned_outcomes(list(runs.values()), expected_identities=expected)
        attempts, metrics, provider_totals, stops, history = [], [], {}, {}, {}
        ledger_index = {entry["attempt_id"]: i + 1 for i, entry in enumerate(ledger["entries"])}
        generation_log = generation.read_json(logs / "generation.log")
        if generation_log.get("identity_hash") != manifest["identity_hash"]:
            raise ValueError("generation log belongs to another campaign")
        for provider, directory in runs.items():
            with generation.run_lock(directory):
                run_manifest, config, records, _, state = generation._load_run(directory)
                by_id, counts = {r["input_id"]: r for r in records}, Counter()
                for attempt in state["attempts"]:
                    counts[attempt["input_id"]] += 1
                    attempts.append(attempt_diagnosis(directory, config, attempt, by_id[attempt["input_id"]],
                                                      counts[attempt["input_id"]], ledger_index[attempt["attempt_id"]]))
                provider_rows = [r for r in planned if r["provider"] == provider]
                metrics.extend(_verify_metrics(directory, run_manifest, state, provider_rows, gt_by_scene))
                selected = [a for a in attempts if a["provider"] == provider]
                provider_totals[provider] = {"planned_inputs": len(provider_rows), "attempts": len(selected),
                    "attempted_inputs": len(counts), "success_inputs": sum(r["status"] == "success" for r in provider_rows),
                    "completed_output_failed_inputs": sum(analysis_v4.attempted_failure(r) for r in provider_rows),
                    "not_run_inputs": sum(r["status"] == "NOT_RUN" for r in provider_rows),
                    "operational_or_unmeasurable_inputs": sum(r["status"] not in {"success", "NOT_RUN"} and not analysis_v4.attempted_failure(r) for r in provider_rows),
                    "retries": sum(a["attempt_number"] > 1 for a in selected),
                    "input_tokens": sum(a["input_tokens"] for a in selected),
                    "output_tokens": sum(a["output_tokens"] for a in selected),
                    "cost_usd_upper_bound": float(sum((Decimal(str(a["cost_usd_upper_bound"])) for a in selected), Decimal(0)))}
                stop_reason = generation_log["providers"][provider].get("generation_stop_reason")
                next_record = next((r for r in records if not any(a["input_id"] == r["input_id"] and a["status"] == "success" for a in state["attempts"])
                                    and counts[r["input_id"]] < config.max_attempts_per_input), None)
                stop = {"recorded_generation_stop_reason": stop_reason, "run_halt_reason": state["halt_reason"],
                        "unattempted_input_ids": [r["input_id"] for r in records if not counts[r["input_id"]]],
                        "next_eligible_input_id": next_record["input_id"] if next_record else None,
                        "next_eligible_scene": next_record["scene_id"] if next_record else None,
                        "next_eligible_condition": next_record["condition"] if next_record else None}
                if stop_reason == "shared_retry_limit":
                    if next_record is None or not counts[next_record["input_id"]] or used["retries"] != manifest["identity"]["budgets"]["max_retries"]:
                        raise ValueError("recorded retry stop does not match frozen attempts and shared budget")
                    stop.update(rejected_attempt_number=counts[next_record["input_id"]] + 1,
                                shared_retries_consumed=used["retries"],
                                scope="provider generation loop returns before later unattempted inputs",
                                inference_basis="saved generation log, ordered inputs, final attempts, validated shared retry total; no reservation executed")
                stops[provider] = stop
                history[provider] = {"requested_model": config.model,
                    "identified_approved_model_response_attempts": sum(a["approved_response_model"] and bool(a["response_id"]) for a in selected),
                    "action_success_inputs": provider_totals[provider]["success_inputs"],
                    "supporting_attempt_ids": [a["attempt_id"] for a in selected if a["approved_response_model"] and a["response_id"]],
                    "historical_evidence_only": True, "current_or_future_availability": "NOT_QUERIED",
                    "frozen_dry_run_availability_display": "NOT_RUN"}
        attempts.sort(key=lambda a: a["ledger_index"])
        if used != generation_log.get("used"):
            raise ValueError("generation log totals differ from authoritative cumulative ledger")
        after = source_inventory(campaign, logs)
        if before != after:
            raise ValueError("original evidence or control content changed during diagnosis")
    return {"attempts": attempts, "planned_outcomes": planned,
            "summary": analysis_v4.summarize_v4(planned, smoke=True), "metric_verification": metrics,
            "ledger_reconciliation": {"campaign_id": manifest["identity"]["campaign_id"],
                "identity_hash": manifest["identity_hash"], "used": used, "providers": provider_totals,
                "research_api_additional_sends": 0, "bill_finalized": False,
                "cost_semantics": "validated ledger conservative upper bound, not finalized billing",
                "budgets": manifest["identity"]["budgets"], "campaign_halt_reason": ledger["halt_reason"],
                "experiment_complete": all(r["status"] == "success" for r in planned), "full_experiment": "NOT_RUN"},
            "stop_diagnosis": stops, "availability_history": history,
            "original_hashes": {"before": before, "after": after, "unchanged": True,
                                "control_files_separate_from_evidence": True}}


def diagnose(campaign_dir, logs_dir, destination, *, gt_by_scene):
    """Create a new diagnostic directory only; never overwrite an existing run."""
    destination = _safe_path(destination)
    for source in (_safe_path(campaign_dir), _safe_path(logs_dir)):
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("diagnostic destination must not overlap original evidence")
    if destination.exists():
        raise FileExistsError("diagnostic destination must be new")
    writable(destination)
    result = build_diagnosis(campaign_dir, logs_dir, gt_by_scene=gt_by_scene)
    destination.mkdir(parents=False, exist_ok=False)
    for name, payload in result.items():
        with (destination / f"{name}.json").open("xb") as stream:
            stream.write(generation.json_bytes(payload))
    for name in ("attempts", "planned_outcomes"):
        rows = result[name]
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (destination / f"{name}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()} for row in rows)
    for path in destination.iterdir():
        scan_file(path)
    return result
