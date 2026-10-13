"""One approved, canonical, append-only 16-request resmoke; no retries.

Reservations precede each POST and survive interruption. The old campaign's
existing locks are held read-only for the entire execution. This is application
coordination, not a claim about OS isolation or a provider's billing guarantee.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from decimal import Decimal
import fcntl
import os
from pathlib import Path
import re
import uuid

from ..server_v2 import providers
from ..server_v2.core import (Limits, ValidationError, evaluate_pair, execute,
                              grid_to_payload, payload_to_grid, strict_json,
                              validate_actions)
from ..server_v2.generation import (GenerationError, atomic_write as _atomic_write, digest,
                                    json_bytes, object_hash, read_json as _read_json,
                                    run_lock as _run_lock, utc_now, write_json as _write_json)
from ..server_v2.io_utils import writable
from . import adapter, schema


CRITICAL_FAILURES = {"auth_failure", "billing_failure", "model_unavailable",
                     "unsupported_schema", "schema_unsupported", "context_failure",
                     "response_identity_failure", "budget_violation"}
APPROVAL_SCOPE = "goal004_fixed16_resmoke_only"


def _checked_path(path):
    """Reject links before resolve(), including links in any parent directory.

Apply to control files, immutable claims, locks and output destinations as well
as response artifacts. The reused IO primitives resolve paths and therefore
cannot themselves distinguish an allowed directory reached through a link.
"""
    candidate = Path(path).absolute()
    if ".." in candidate.parts or any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise GenerationError("symlink_or_traversal_path_forbidden")
    return writable(candidate)


def read_json(path):
    return _read_json(_checked_path(path))


def write_json(path, value):
    _write_json(_checked_path(path), value)


def atomic_write(path, data):
    _atomic_write(_checked_path(path), data)


@contextmanager
def run_lock(directory):
    directory = _checked_path(directory)
    _checked_path(directory / "run.lock")
    with _run_lock(directory):
        yield


def _plan(path):
    from .plan import load_plan
    return load_plan(path)


def _verify_parent(identity):
    from .plan import verify_parent
    verify_parent(identity["parent"])


def _require_current_pricing(identity):
    today = utc_now()[:10]
    if any(record.get("verified") is not True or record.get("verified_on") != today
           for record in identity["pricing"].values()):
        raise GenerationError("pricing_documentation_requires_today_verification_and_replan")


@contextmanager
def _parent_locked(identity):
    """Do not create, truncate, rewrite, or touch the parent lock contents."""
    directory = _checked_path(identity["parent"]["campaign_directory"])
    campaign = read_json(directory / "campaign.json")
    paths = [directory / "run.lock", directory / "budget.lock"]
    paths += [directory / campaign["identity"]["runs"][p]["path"] / "run.lock"
              for p in sorted(campaign["identity"]["runs"])]
    with ExitStack() as stack:
        for path in paths:
            stream = stack.enter_context(_checked_path(path).open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError("parent_generation_locked") from None
            stack.callback(fcntl.flock, stream, fcntl.LOCK_UN)
        _verify_parent(identity)
        yield


@contextmanager
def _authority_locked(identity):
    path = _checked_path(identity["authority_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with _checked_path(path.with_suffix(path.suffix + ".lock")).open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GenerationError("lineage_activation_locked") from None
        try:
            yield path
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _approval(plan, approval):
    expected = {"manual_confirmed", "plan_identity_hash", "enabled_directory",
                "approved_at_utc", "confirmation_channel", "scope"}
    if (not isinstance(approval, dict) or set(approval) != expected
            or approval["manual_confirmed"] is not True
            or approval["plan_identity_hash"] != plan["identity_hash"]
            or approval["enabled_directory"] != plan["identity"]["enabled_directory"]
            or approval["confirmation_channel"] != "/dev/tty"
            or approval["scope"] != APPROVAL_SCOPE
            or not isinstance(approval["approved_at_utc"], str)
            or approval["approved_at_utc"][:10] != utc_now()[:10]):
        raise GenerationError("fresh_exact_scope_tty_approval_required")


def _authority(plan, approval):
    return {"version": "resmoke_v2_authority_v1", "plan_identity_hash": plan["identity_hash"],
            "enabled_directory": plan["identity"]["enabled_directory"],
            "approval_hash": object_hash(approval)}


def initialize_enabled(plan_path, *, approval: dict) -> Path:
    """Called only after the launcher's exact /dev/tty approval, or fixtures.

The lineage claim is persisted before enabled files. Interrupted activation
does not permit another plan/directory to mint a fresh budget.
"""
    plan = _plan(plan_path)
    identity = plan["identity"]
    _approval(plan, approval)
    _require_current_pricing(identity)
    directory = _checked_path(identity["enabled_directory"])
    with _parent_locked(identity), _authority_locked(identity) as authority_path:
        if authority_path.exists():
            if read_json(authority_path) != _authority(plan, approval):
                raise GenerationError("parent_lineage_already_claimed; resume_original_ledger")
            raise GenerationError("lineage_activation_already_started; resume_original_ledger")
        if directory.exists():
            raise GenerationError("enabled_directory_already_exists; never_reset_budget")
        write_json(authority_path, _authority(plan, approval))
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "approval.json", approval)
        write_json(directory / "enabled.json", {"api_allowed": True,
                   "plan_identity_hash": plan["identity_hash"], "approval_hash": object_hash(approval),
                   "enabled_directory": str(directory), "protocol_id": identity["protocol_id"]})
        write_json(directory / "ledger.json", {"version": "resmoke_v2_ledger_v1",
                   "plan_identity_hash": plan["identity_hash"], "entries": [], "halt_reason": None})
    return directory


def _artifact(directory, descriptor):
    relative = Path(descriptor["path"])
    candidate = directory / relative
    path = candidate.resolve()
    if (relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(directory)
            or any(p.is_symlink() for p in (candidate, *candidate.parents))):
        raise GenerationError("invalid_artifact_path")
    data = path.read_bytes()
    if digest(data) != descriptor["sha256"] or len(data) != descriptor["bytes"]:
        raise GenerationError("artifact_hash_mismatch")
    return path


def _reservation(identity, record):
    pricing = identity["pricing"][record["provider"]]
    input_tokens = pricing["context_window_tokens"]
    output_tokens = identity["budgets"]["max_output_tokens"]
    return input_tokens, output_tokens, providers.cost_ceiling(input_tokens, output_tokens, pricing)


def _payload(identity, record):
    return adapter.build_payload(record["provider"], record["model"], identity["prompt"],
                                 record["input_text"], identity["budgets"]["max_output_tokens"])


def _claim(plan, entry):
    return {"plan_identity_hash": plan["identity_hash"], **{name: entry[name] for name in (
        "attempt_id", "schedule_index", "input_id", "provider", "request_payload_hash")}}


def _validate_response_artifacts(directory, identity, record, entry):
    artifacts = entry["artifacts"]
    received = "raw.json" in artifacts or entry.get("http_status_code") is not None
    if entry.get("api_response_received") is not received:
        raise GenerationError("response_presence_mismatch")
    if "raw.json" not in artifacts:
        return
    raw = _artifact(directory, artifacts["raw.json"]).read_bytes()
    if entry.get("response_parsed") is not True:
        if entry["usage"] is not None or entry["status"] == "success":
            raise GenerationError("unparsed_response_has_known_success_metadata")
        return
    try:
        parsed = adapter.parse_response(record["provider"], raw)
    except providers.ProviderError:
        if entry["usage"] is not None or entry["status"] == "success":
            raise GenerationError("invalid_raw_response_has_known_success_metadata") from None
        return
    if (entry.get("response_id") != parsed.response_id or entry.get("response_model") != parsed.model
            or entry.get("natural_stop") is not parsed.complete
            or (entry["usage"] is not None and entry["usage"] != parsed.usage)):
        raise GenerationError("raw_response_metadata_mismatch")
    if entry["status"] == "success":
        if not parsed.complete or parsed.failure_kind:
            raise GenerationError("unsuccessful_raw_response_recorded_as_success")
        limits = Limits(**identity.get("limits", {}))
        actions = validate_actions(schema.validate_document(strict_json(parsed.text, limits)), limits)
        grid = execute(actions, limits)
        if (read_json(_artifact(directory, artifacts["parsed.json"])) != {"operations": actions}
                or read_json(_artifact(directory, artifacts["voxels.json"])) != grid_to_payload(grid)
                or entry.get("operation_count") != len(actions)
                or entry.get("final_voxel_count") != len(grid.voxels)
                or entry.get("empty_output") is not (not bool(grid.voxels))
                or any(entry.get(name) is not True for name in (
                    "json_success", "schema_success", "executor_success"))):
            raise GenerationError("successful_execution_artifact_mismatch")


def _load_enabled(plan):
    identity = plan["identity"]
    directory = _checked_path(identity["enabled_directory"])
    for name in ("attempts", "reservations", "evaluation.json", "summary.json", "run.lock"):
        _checked_path(directory / name)
    approval = read_json(directory / "approval.json")
    # Historical approval dates remain valid for a same-ledger resume.
    expected = {"api_allowed": True, "plan_identity_hash": plan["identity_hash"],
                "approval_hash": object_hash(approval), "enabled_directory": str(directory),
                "protocol_id": identity["protocol_id"]}
    if read_json(directory / "enabled.json") != expected:
        raise GenerationError("enabled_scope_mismatch")
    if (approval.get("manual_confirmed") is not True
            or approval.get("plan_identity_hash") != plan["identity_hash"]
            or approval.get("enabled_directory") != str(directory)
            or approval.get("confirmation_channel") != "/dev/tty"
            or approval.get("scope") != APPROVAL_SCOPE):
        raise GenerationError("enabled_approval_mismatch")
    if read_json(Path(identity["authority_path"])) != _authority(plan, approval):
        raise GenerationError("parent_lineage_authority_mismatch")
    ledger = read_json(directory / "ledger.json")
    if ledger.get("plan_identity_hash") != plan["identity_hash"]:
        raise GenerationError("ledger_identity_mismatch")
    entries = ledger.get("entries")
    schedule = identity["schedule"]
    if not isinstance(entries, list) or len(entries) > len(schedule):
        raise GenerationError("invalid_ledger_schedule")
    claims = list(_checked_path(directory / "reservations").glob("*.json"))
    if len(claims) != len(entries):
        raise GenerationError("reservation_claim_ledger_mismatch; partial_restore_or_interrupted_reservation")
    attempt_ids = set()
    for index, entry in enumerate(entries):
        record = schedule[index]
        if (entry.get("schedule_index") != index or entry.get("input_id") != record["input_id"]
                or entry.get("provider") != record["provider"] or entry.get("retry") is not False
                or entry.get("status") not in {"reserved", "success", "failed"}
                or not isinstance(entry.get("attempt_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", entry["attempt_id"])
                or entry.get("attempt_id") in attempt_ids
                or entry.get("request_payload_hash") != object_hash(_payload(identity, record))):
            raise GenerationError("invalid_ledger_attempt_scope")
        attempt_ids.add(entry["attempt_id"])
        claim_path = directory / "reservations" / (entry["attempt_id"] + ".json")
        if read_json(claim_path) != _claim(plan, entry):
            raise GenerationError("immutable_reservation_claim_mismatch")
        reserved = _reservation(identity, record)
        if entry.get("usage") is None:
            expected_charge = reserved
            if entry["status"] == "success":
                raise GenerationError("unknown_usage_cannot_be_success")
        else:
            expected_charge = providers.usage_upper_cost(record["provider"], entry["usage"],
                                                         identity["pricing"][record["provider"]])
        actual = (entry["charged_input_tokens"], entry["charged_output_tokens"], entry["charged_cost_usd"])
        if actual != expected_charge:
            raise GenerationError("ledger_billing_mismatch")
        for name, descriptor in entry["artifacts"].items():
            if (name not in {"raw.json", "parsed.json", "voxels.json"}
                    or descriptor["path"] != f"attempts/{entry['attempt_id']}/{name}"):
                raise GenerationError("unapproved_attempt_artifact")
            _artifact(directory, descriptor)
        if entry["status"] == "success":
            if not {"raw.json", "parsed.json", "voxels.json"} <= entry["artifacts"].keys():
                raise GenerationError("success_artifacts_missing")
            if not _completed_response(identity, record, entry):
                raise GenerationError("success_response_identity_mismatch")
        _validate_response_artifacts(directory, identity, record, entry)
        if entry.get("failure_kind") in CRITICAL_FAILURES and not ledger.get("halt_reason"):
            raise GenerationError("critical_halt_missing")
        if index < len(entries) - 1 and (entry["status"] == "reserved" or entry.get("usage") is None):
            raise GenerationError("continued_after_uncertain_attempt")
    return directory, ledger


def budget_totals(identity, ledger):
    entries = ledger["entries"]
    parent = identity["parent"]["usage"]
    cost = sum((Decimal(str(e["charged_cost_usd"])) for e in entries), Decimal(0))
    tokens = sum(e["charged_input_tokens"] + e["charged_output_tokens"] for e in entries)
    return {"prior": parent, "new": {"calls": len(entries), "retries": 0, "tokens": tokens,
            "cost_usd_upper_bound": float(cost),
            "unknown_usage_attempts": sum(e["usage"] is None for e in entries)},
            "combined": {"calls": parent["calls"] + len(entries), "retries": parent["retries"],
                         "tokens": parent["tokens"] + tokens,
                         "cost_usd_upper_bound": float(Decimal(str(parent["cost_usd_upper_bound"])) + cost)}}


def _can_reserve(identity, ledger, reservation):
    used = budget_totals(identity, ledger)
    budgets = identity["budgets"]
    tokens = reservation[0] + reservation[1]
    combined_cost = (Decimal(str(identity["parent"]["usage"]["cost_usd_upper_bound"]))
                     + sum((Decimal(str(e["charged_cost_usd"])) for e in ledger["entries"]), Decimal(0))
                     + Decimal(str(reservation[2])))
    return (used["new"]["calls"] + 1 <= budgets["max_new_calls"]
            and used["combined"]["calls"] + 1 <= budgets["max_total_calls"]
            and used["new"]["tokens"] + tokens <= budgets["max_tokens"]
            and used["combined"]["tokens"] + tokens <= budgets["max_total_tokens"]
            and combined_cost <= Decimal(str(budgets["max_cost_usd"])))


def _save_artifact(directory, entry, name, data):
    relative = Path("attempts") / entry["attempt_id"] / name
    target = _checked_path(directory / relative)
    if target.exists():
        raise GenerationError("attempt_artifact_already_exists")
    atomic_write(target, data)
    entry["artifacts"][name] = {"path": relative.as_posix(), "sha256": digest(data), "bytes": len(data)}


def _completed_response(identity, record, entry):
    return bool(entry and entry.get("response_id") and entry.get("usage") is not None
                and "raw.json" in entry.get("artifacts", {})
                and entry.get("response_model") in providers.accepted_response_model_ids(
                    record["provider"], record["model"], identity["pricing"][record["provider"]]))


def execution_summary(plan, ledger):
    entries = ledger["entries"]
    complete = (len(entries) == len(plan["identity"]["schedule"])
                and all(e["status"] != "reserved" and e["usage"] is not None for e in entries)
                and ledger["halt_reason"] is None)
    return {"protocol_id": plan["identity"]["protocol_id"], "plan_identity_hash": plan["identity_hash"],
            "experiment_complete": complete, "all_inputs_attempted": len(entries) == 16,
            "status": "COMPLETE_FIXED16" if complete else "PARTIAL" if entries else "NOT_RUN",
            "planned_inputs": 16, "attempted_inputs": len(entries), "not_run_inputs": 16 - len(entries),
            "halt_reason": ledger["halt_reason"], "used": budget_totals(plan["identity"], ledger),
            "full_experiment_960": "NOT_RUN", "additional_retries": 0}


def run(plan_path, *, permission: bool = False, transports: dict | None = None) -> dict:
    """Execute the entire fixed interleaved schedule, never a provider subset."""
    if permission is not True:
        raise GenerationError("explicit_paid_resmoke_invocation_permission_required")
    plan = _plan(plan_path)
    identity = plan["identity"]
    _require_current_pricing(identity)
    if transports is not None and (identity["provenance"] != "fixture_only"
                                  or set(transports) != {"openai", "anthropic"}):
        raise GenerationError("both_fixture_transports_require_fixture_only_plan")
    directory = _checked_path(identity["enabled_directory"])
    if not directory.is_dir():
        raise GenerationError("run_not_enabled; human_launcher_approval_required")
    with _parent_locked(identity), run_lock(directory), _authority_locked(identity):
        directory, ledger = _load_enabled(plan)
        if any(e["status"] == "reserved" or e["usage"] is None for e in ledger["entries"]):
            ledger["halt_reason"] = ledger["halt_reason"] or "unreconciled_prior_reservation"
            write_json(directory / "ledger.json", ledger)
        if ledger["halt_reason"]:
            return execution_summary(plan, ledger)
        if not all(bool(os.environ.get(key)) for key in providers.KEY_NAMES.values()):
            raise GenerationError("both_provider_keys_required_before_any_request")
        limits = Limits(**identity.get("limits", {}))
        for index in range(len(ledger["entries"]), len(identity["schedule"])):
            record = identity["schedule"][index]
            provider = record["provider"]
            pricing = identity["pricing"][provider]
            reservation = _reservation(identity, record)
            if not _can_reserve(identity, ledger, reservation):
                ledger["halt_reason"] = "budget_insufficient"
                write_json(directory / "ledger.json", ledger)
                break
            payload = _payload(identity, record)
            entry = {"attempt_id": uuid.uuid4().hex, "schedule_index": index,
                     "input_id": record["input_id"], "provider": provider, "retry": False,
                     "request_payload_hash": object_hash(payload), "started_at_utc": utc_now(),
                     "status": "reserved", "failure_kind": None, "failure_stage": None,
                     "usage": None, "response_id": None, "response_model": None,
                     "api_response_received": False, "response_parsed": False,
                     "http_status_code": None, "natural_stop": None,
                     "json_success": None, "schema_success": None, "executor_success": None,
                     "operation_count": None, "final_voxel_count": None, "empty_output": None,
                     "charged_input_tokens": reservation[0], "charged_output_tokens": reservation[1],
                     "charged_cost_usd": reservation[2], "billing": "worst_case_reservation",
                     "artifacts": {}}
            for name in ("raw.json", "parsed.json", "voxels.json"):
                _checked_path(directory / "attempts" / entry["attempt_id"] / name)
            ledger["entries"].append(entry)
            # A second immutable record detects a rolled-back mutable ledger.
            # A crash between claim and ledger fails closed; it never resends.
            write_json(directory / "reservations" / (entry["attempt_id"] + ".json"), _claim(plan, entry))
            write_json(directory / "ledger.json", ledger)
            try:
                raw = adapter.send_request(provider, payload, api_allowed=True, permission=True,
                                           timeout=identity.get("timeout_seconds", 120.0),
                                           transport=(transports or {}).get(provider))
                if not isinstance(raw, bytes):
                    raise providers.ProviderError("response_parse_failure")
                entry["api_response_received"] = True
                _save_artifact(directory, entry, "raw.json", raw)
                response = adapter.parse_response(provider, raw)
                entry["response_parsed"] = True
                entry["response_id"], entry["response_model"] = response.response_id, response.model
                entry["natural_stop"] = response.complete
                charge = providers.usage_upper_cost(provider, response.usage, pricing)
                entry["usage"] = response.usage
                entry["charged_input_tokens"], entry["charged_output_tokens"], entry["charged_cost_usd"] = charge
                entry["billing"] = "known_usage_conservative_cost"
                if any(a > b for a, b in zip(charge, reservation)):
                    raise providers.ProviderError("budget_violation")
                if not _completed_response(identity, record, entry):
                    raise providers.ProviderError("response_identity_failure")
                if response.failure_kind:
                    raise providers.ProviderError(response.failure_kind)
                if not response.complete:
                    raise providers.ProviderError("incomplete_response")
                try:
                    document = strict_json(response.text, limits)
                    entry["json_success"] = True
                except (ValidationError, TypeError):
                    entry["json_success"], entry["failure_stage"] = False, "json"
                    raise providers.ProviderError("parse_failure") from None
                try:
                    actions = schema.validate_document(document)
                    entry["schema_success"] = True
                except ValidationError:
                    entry["schema_success"], entry["failure_stage"] = False, "schema"
                    raise providers.ProviderError("validation_failure") from None
                try:
                    actions = validate_actions(actions, limits)
                except ValidationError:
                    entry["failure_stage"] = "local_validator"
                    raise providers.ProviderError("validation_failure") from None
                entry["operation_count"] = len(actions)
                _save_artifact(directory, entry, "parsed.json", json_bytes({"operations": actions}))
                try:
                    grid = execute(actions, limits)
                except (ValidationError, ValueError):
                    entry["executor_success"], entry["failure_stage"] = False, "executor"
                    raise providers.ProviderError("execution_failure") from None
                entry["executor_success"] = True
                entry["final_voxel_count"] = len(grid.voxels)
                entry["empty_output"] = not bool(grid.voxels)
                _save_artifact(directory, entry, "voxels.json", json_bytes(grid_to_payload(grid)))
                entry["status"] = "success"
            except providers.ProviderError as exc:
                entry["http_status_code"] = exc.status_code
                if exc.status_code is not None:
                    entry["api_response_received"] = True
                # Some future narrow HTTP adapter may expose a safe raw body.
                # Existing adapters suppress bodies, so this remains optional.
                raw_error = getattr(exc, "raw", None)
                if isinstance(raw_error, bytes) and "raw.json" not in entry["artifacts"]:
                    entry["api_response_received"] = True
                    _save_artifact(directory, entry, "raw.json", raw_error)
                entry["status"], entry["failure_kind"] = "failed", exc.kind
                if entry["usage"] is None:
                    ledger["halt_reason"] = "unknown_usage:" + exc.kind
                elif exc.kind in CRITICAL_FAILURES:
                    ledger["halt_reason"] = exc.kind
            entry["finished_at_utc"] = utc_now()
            write_json(directory / "ledger.json", ledger)
            if ledger["halt_reason"]:
                break
        return execution_summary(plan, ledger)


def evaluate_and_summarize(plan_path, gt_by_scene=None) -> dict:
    """CPU only. Empty successful executions keep their actual zero quality."""
    from ..server_v2.analysis_v4 import summarize_v4
    from ..server_v2.alignment import AlignmentResourceLimit
    from ..server_v2.data import load_scene

    plan = _plan(plan_path)
    identity = plan["identity"]
    directory = _checked_path(identity["enabled_directory"])
    with run_lock(directory):
        directory, ledger = _load_enabled(plan)
        grids = {} if gt_by_scene is None else dict(gt_by_scene)
        rows = []
        for index, record in enumerate(identity["schedule"]):
            entry = ledger["entries"][index] if index < len(ledger["entries"]) else None
            row = {"provider": record["provider"], "scene_id": record["scene_id"],
                   "condition": record["condition"], "description_model": record.get("description_model"),
                   "provenance": "new_implementation", "protocol_id": identity["protocol_id"],
                   "input_id": record["input_id"], "attempt_id": entry["attempt_id"] if entry else None,
                   "attempted": entry is not None, "status": entry["status"] if entry else "NOT_RUN",
                   "failure_kind": entry["failure_kind"] if entry else None,
                   "completed_request_outcome": _completed_response(identity, record, entry),
                   "metrics": None, "valid_nonempty_reconstruction": False}
            for name in ("api_response_received", "natural_stop", "json_success", "schema_success",
                         "executor_success", "operation_count", "final_voxel_count", "empty_output"):
                row[name] = entry.get(name) if entry else None
            if entry and entry["status"] == "success":
                descriptor = identity["gt_by_scene"][record["scene_id"]]
                if record["scene_id"] not in grids:
                    grids[record["scene_id"]] = load_scene(descriptor["building_directory"]).grid
                gt = grids[record["scene_id"]]
                if object_hash(grid_to_payload(gt)) != descriptor["grid_sha256"]:
                    raise GenerationError("evaluation_gt_hash_mismatch")
                grid = payload_to_grid(read_json(_artifact(directory, entry["artifacts"]["voxels.json"])))
                row["valid_nonempty_reconstruction"] = bool(grid.voxels)
                try:
                    row["metrics"] = evaluate_pair(gt, grid, alignment_mode="strict_translation")
                except AlignmentResourceLimit as exc:
                    row["status"] = "ALIGNMENT_RESOURCE_LIMIT"
                    row["evaluation_failure"] = {"kind": "ALIGNMENT_RESOURCE_LIMIT", "metadata": exc.metadata}
            rows.append(row)
        summary = summarize_v4(rows, smoke=True)
        summary.update(execution_summary(plan, ledger))
        summary["stage_counts"] = {name: sum(row[name] is True for row in rows) for name in (
            "api_response_received", "natural_stop", "json_success", "schema_success", "executor_success",
            "empty_output", "valid_nonempty_reconstruction", "completed_request_outcome")}
        summary["quality_evaluated_inputs"] = sum(row["metrics"] is not None for row in rows)
        summary["account_native_format_acceptance"] = "NOT_RUN" if not ledger["entries"] else (
            "FIXTURE_ONLY" if identity["provenance"] == "fixture_only" else "SEE_INDIVIDUAL_RESPONSES")
        write_json(directory / "evaluation.json", {"plan_identity_hash": plan["identity_hash"], "rows": rows})
        write_json(directory / "summary.json", summary)
        return summary
