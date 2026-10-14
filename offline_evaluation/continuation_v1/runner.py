"""Canonical suffix800 continuation; old payloads and pure accounting are reused.

Only run scope, parent lineage, indices and checkpoint coordination are new.
Reservations are immutable before POST. No retry or copied-budget activation.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import uuid

from ..batch_v1 import runner as base
from ..server_v2 import providers
from ..server_v2.core import Limits, ValidationError, execute, grid_to_payload, strict_json, validate_actions
from ..server_v2.generation import GenerationError, digest, json_bytes, object_hash, utc_now
from ..resmoke_v2 import adapter, schema

CRITICAL_FAILURES = base.CRITICAL_FAILURES
APPROVAL_SCOPE = 'goal006_fixed_evaluation_suffix800_only_cumulative_usd30'
_checked_path = base._checked_path
read_json = base.read_json
write_json = base.write_json
atomic_write = base.atomic_write
run_lock = base.run_lock
_authority_locked = base._authority_locked
_artifact = base._artifact
_reservation = base._reservation
_payload = base._payload
_validate_response_artifacts = base._validate_response_artifacts
budget_totals = base.budget_totals
_can_reserve = base._can_reserve
_save_artifact = base._save_artifact
_completed_response = base._completed_response


def _plan(path):
    from .plan import load_plan
    return load_plan(path)


def _verify_parent(identity):
    from .plan import verify_parent
    verify_parent(identity['parent'])


def _require_current_pricing(identity):
    from .plan import current_pricing
    return current_pricing(identity, fresh=True)


@contextmanager
def _parent_locked(identity):
    """Hold old campaign and completed resmoke locks without writing them."""
    with ExitStack() as stack:
        for path in identity["parent"]["lock_paths"]:
            stream = stack.enter_context(_checked_path(path).open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError("parent_generation_locked") from None
            stack.callback(fcntl.flock, stream, fcntl.LOCK_UN)
        _verify_parent(identity)
        yield


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
    return {"version": "continuation_v1_authority_v1", "plan_identity_hash": plan["identity_hash"],
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
        write_json(directory / "ledger.json", {"version": "continuation_v1_ledger_v1",
                   "plan_identity_hash": plan["identity_hash"], "entries": [], "halt_reason": None})
    return directory


def _claim(plan, entry):
    return {"plan_identity_hash": plan["identity_hash"], **{name: entry[name] for name in (
        "attempt_id", "local_index", "schedule_index", "input_id", "provider", "request_payload_hash")}}


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
    if (ledger.get("plan_identity_hash") != plan["identity_hash"]
            or ledger.get("version") != "continuation_v1_ledger_v1"):
        raise GenerationError("ledger_identity_mismatch")
    entries = ledger.get("entries")
    schedule = identity["schedule"]
    if not isinstance(entries, list) or len(entries) > len(schedule):
        raise GenerationError("invalid_ledger_schedule")
    claims = list(_checked_path(directory / "reservations").glob("*.json"))
    if len(claims) != len(entries):
        raise GenerationError("reservation_claim_ledger_mismatch; partial_restore_or_interrupted_reservation")
    attempt_ids = set()
    semantic_pairs = set()
    for index, entry in enumerate(entries):
        record = schedule[index]
        if (entry.get("local_index") != index or entry.get("schedule_index") != record["schedule_index"] or entry.get("input_id") != record["input_id"]
                or entry.get("provider") != record["provider"] or entry.get("retry") is not False
                or entry.get("status") not in {"reserved", "success", "failed"}
                or not isinstance(entry.get("attempt_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", entry["attempt_id"])
                or entry.get("attempt_id") in attempt_ids
                or entry.get("request_payload_hash") != object_hash(_payload(identity, record))):
            raise GenerationError("invalid_ledger_attempt_scope")
        pair = (entry["provider"], entry["input_id"])
        if pair in semantic_pairs:
            raise GenerationError("duplicate_provider_input_attempt")
        semantic_pairs.add(pair)
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


def execution_summary(plan, ledger):
    entries = ledger["entries"]
    complete = (len(entries) == len(plan["identity"]["schedule"])
                and all(e["status"] != "reserved" and e["usage"] is not None for e in entries)
                and ledger["halt_reason"] is None)
    return {"protocol_id": plan["identity"]["protocol_id"], "plan_identity_hash": plan["identity_hash"],
            "experiment_complete": complete, "all_inputs_attempted": len(entries) == len(plan["identity"]["schedule"]),
            "status": "COMPLETE_CONTINUATION800" if complete else "PARTIAL" if entries else "NOT_RUN",
            "planned_inputs": len(plan["identity"]["schedule"]), "attempted_inputs": len(entries),
            "not_run_inputs": len(plan["identity"]["schedule"]) - len(entries),
            "halt_reason": ledger["halt_reason"], "used": budget_totals(plan["identity"], ledger),
            "full_experiment_960": "COMPLETE_FIXED960" if complete else "INTERIM_DESCRIPTIVE",
            "semantic_protocol_id": plan["identity"]["semantic_protocol_id"],
            "completed_parent_inputs": 160,
            "additional_retries": 0}


def run(plan_path, *, permission: bool = False, transports: dict | None = None,
        checkpoint_callback=None, github_checkpoints=False) -> dict:
    """Execute the entire fixed interleaved schedule, never a provider subset."""
    if permission is not True:
        raise GenerationError("explicit_paid_continuation_invocation_permission_required")
    plan = _plan(plan_path)
    identity = plan["identity"]
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
            saved = _checkpoint(plan, directory, ledger, "existing_halt", checkpoint_callback, github_checkpoints)
            return _result(plan, ledger, checkpoint=saved)
        if execution_summary(plan, ledger)["experiment_complete"]:
            return _result(plan, ledger)
        if not all(bool(os.environ.get(key)) for key in providers.KEY_NAMES.values()):
            raise GenerationError("both_provider_keys_required_before_any_request")
        limits = Limits(**identity.get("limits", {}))
        for index in range(len(ledger["entries"]), len(identity["schedule"])):
            # Recheck UTC freshness at every new send, including a midnight crossing.
            try:
                _require_current_pricing(identity)
            except GenerationError as exc:
                saved = _checkpoint(plan, directory, ledger, "pricing_freshness_stop", checkpoint_callback, github_checkpoints)
                return _result(plan, ledger, operational_stop=str(exc), checkpoint=saved)
            record = identity["schedule"][index]
            provider = record["provider"]
            pricing = identity["pricing"][provider]
            reservation = _reservation(identity, record)
            if not _can_reserve(identity, ledger, reservation):
                ledger["halt_reason"] = "budget_insufficient"
                write_json(directory / "ledger.json", ledger)
                break
            payload = _payload(identity, record)
            entry = {"attempt_id": uuid.uuid4().hex, "local_index": index, "schedule_index": record["schedule_index"],
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
            _progress(plan, ledger)
            if ledger["halt_reason"]:
                break
            if (index + 1) % 160 == 0:
                saved = _checkpoint(plan, directory, ledger, "segment_complete", checkpoint_callback, github_checkpoints)
                if saved.get("status") not in {"SAVED", "LOCAL_SAVED", "LOCAL_VERIFIED", "GITHUB_VERIFIED", "GITHUB_VERIFIED_DRAFT", "FIXTURE_SAVED"}:
                    return _result(plan, ledger, operational_stop="checkpoint_save_failed", checkpoint=saved)
        if ledger["halt_reason"]:
            saved = _checkpoint(plan, directory, ledger, "safe_halt", checkpoint_callback, github_checkpoints)
            return _result(plan, ledger, checkpoint=saved)
        return _result(plan, ledger)


def resume_command(plan, *, github=True):
    from ..server_v2.io_utils import ROOT
    command = [str(ROOT / '.venv-server-v2-20261013/bin/python'), '-B',
               str(ROOT / 'offline_evaluation/continuation_v1/launch.py'),
               '--plan', plan['identity']['plan_path']]
    if github:
        command.append('--github-checkpoints')
    return shlex.join(command)


def _result(plan, ledger, **extra):
    return {**execution_summary(plan, ledger), **extra, 'resume_command': resume_command(plan)}


def _progress(plan, ledger):
    last = ledger['entries'][-1] if ledger['entries'] else None
    print(json.dumps({'progress': f"{len(ledger['entries'])}/{len(plan['identity']['schedule'])}",
          'status': last['status'] if last else 'NOT_RUN',
          'usage': budget_totals(plan['identity'], ledger),
          'output_directory': plan['identity']['enabled_directory']}, sort_keys=True), flush=True)


def _checkpoint(plan, directory, ledger, reason, callback, github):
    try:
        if callback is not None:
            return callback(plan, directory, ledger, reason, github=github)
        from .aggregate import evaluate_verified
        from .checkpoint import safe_checkpoint
        evaluate_verified(plan, directory, ledger)
        return safe_checkpoint(plan, directory, ledger, reason, github=github)
    except Exception as exc:
        # Saving failure is operational, never a fabricated generation failure.
        # Results and reservations stay on disk; checkpoint retry never calls run.
        return {'status': 'SAVE_FAILED_RETRY_AVAILABLE', 'error_type': type(exc).__name__,
                'reason': reason, 'generation_ledger_unchanged_by_save_failure': True}


def evaluate_and_summarize(plan_path, gt_by_scene=None):
    """CPU-only incremental evaluation; disabled plans never create enabled state."""
    from .aggregate import evaluate_verified
    plan = _plan(plan_path)
    directory = _checked_path(plan['identity']['enabled_directory'])
    if not directory.is_dir():
        raise GenerationError('run_not_enabled; CPU evaluation requires an existing approved run')
    with run_lock(directory):
        directory, ledger = _load_enabled(plan)
        return evaluate_verified(plan, directory, ledger, gt_by_scene=gt_by_scene)


def status(plan_path):
    """Read-only and independent of UTC pricing freshness."""
    plan = _plan(plan_path)
    identity = plan['identity']
    _verify_parent(identity)
    directory = _checked_path(identity['enabled_directory'])
    if not directory.exists():
        authority = _checked_path(identity['authority_path'])
        if authority.exists():
            raise GenerationError('partial_activation_authority_without_enabled; never_reset')
        return _result(plan, {'entries': [], 'halt_reason': None}, approved=False)
    # Open an already existing lock, without creating any run/control files.
    lock_path = directory / 'run.lock'
    if not lock_path.exists():
        authority = _checked_path(identity['authority_path'])
        lock_path = authority.with_suffix(authority.suffix + '.lock')
    with _checked_path(lock_path).open('rb') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GenerationError('generation_active_status_snapshot_unavailable') from None
        try:
            _, ledger = _load_enabled(plan)
            return _result(plan, ledger, approved=True)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
