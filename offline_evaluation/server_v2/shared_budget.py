"""One durable smoke budget across providers, runs, retries, and restarts.

The campaign identity and approval are immutable; every reservation is fsynced
before a request. A campaign lock covers the request and reconciliation, so
concurrency is one. Uncertain billing keeps its full reservation and stops both
providers. This is an application scope restriction, not OS isolation.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import fcntl
import math
from pathlib import Path
import re

from .generation import GenerationError, object_hash, read_json, utc_now, write_json
from .io_utils import writable


class BudgetStop(GenerationError):
    pass


CRITICAL_FAILURES = {"auth_failure", "billing_failure", "model_unavailable", "context_failure", "response_identity_failure", "budget_violation"}


def validate_budgets(budgets: dict) -> dict:
    required = {"max_calls", "max_retries", "max_attempts_per_input", "max_cost_usd", "max_tokens", "max_output_tokens", "concurrency"}
    if not isinstance(budgets, dict) or set(budgets) != required:
        raise GenerationError("invalid shared budget fields")
    for name in required - {"max_cost_usd"}:
        if type(budgets[name]) is not int or budgets[name] < 0:
            raise GenerationError("invalid shared budget integer")
    if not 1 <= budgets["max_calls"] <= 20 or not 0 <= budgets["max_retries"] <= 4:
        raise GenerationError("smoke call/retry caps exceed proposal")
    if budgets["max_attempts_per_input"] != 2 or budgets["concurrency"] != 1:
        raise GenerationError("smoke requires two attempts maximum and concurrency one")
    if budgets["max_tokens"] <= 0 or budgets["max_output_tokens"] != 16_384:
        raise GenerationError("explicit global token ceiling and output16384 required")
    cost = budgets["max_cost_usd"]
    if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 < cost <= 5:
        raise GenerationError("smoke USD ceiling must be positive and at most5")
    return dict(budgets)


def load_campaign(directory) -> tuple[dict, dict]:
    directory = writable(directory)
    manifest = read_json(directory / "campaign.json")
    identity = manifest["identity"]
    if identity.get("ledger_directory") != str(directory):
        raise GenerationError("campaign_ledger_directory_mismatch; resume the approved original ledger")
    if manifest.get("identity_hash") != object_hash(identity):
        raise GenerationError("campaign_identity_mismatch")
    validate_budgets(identity["budgets"])
    from .generation import _implementation_hashes
    if identity.get("implementation_hashes") != _implementation_hashes():
        raise GenerationError("campaign_implementation_mismatch")
    if manifest.get("api_allowed") is not True:
        raise GenerationError("campaign_api_disabled")
    approval = read_json(directory / "approval.json")
    if object_hash(approval) != identity.get("approval_hash") or approval.get("manual_confirmed") is not True:
        raise GenerationError("campaign_approval_mismatch")
    if approval.get("proposal_identity_hash") != identity.get("proposal_identity_hash"):
        raise GenerationError("campaign_approval_scope_mismatch")
    if (identity["budgets"] != approval.get("budgets") or identity["targets"] != approval.get("target_inputs")
        or identity["campaign_id"] != approval.get("campaign_id")
        or identity["ledger_directory"] != approval.get("enabled_directory")
        or identity["implementation_hashes"] != approval.get("implementation_hashes")):
        raise GenerationError("campaign_approval_scope_mismatch")
    from .generation import _load_run
    local_attempts = {}
    for provider, run in identity["runs"].items():
        run_manifest, config, _, _, state = _load_run(directory / run["path"])
        if (object_hash(run_manifest["identity"]) != run["identity_hash"]
            or run_manifest["identity"]["config_hash"] != approval["enabled_config_hashes"].get(provider)
            or run_manifest["identity"]["prompt_hash"] != approval["prompt_hashes"].get(provider)
            or config.model != approval["models"].get(provider) or not config.api_allowed
            or config.budget_campaign_id != identity["campaign_id"]):
            raise GenerationError("campaign_run_approval_mismatch")
        for attempt in state["attempts"]:
            if attempt["attempt_id"] in local_attempts:
                raise GenerationError("duplicate_local_campaign_attempt_id")
            local_attempts[attempt["attempt_id"]] = provider, attempt
    ledger = read_json(directory / "ledger.json")
    if ledger.get("identity_hash") != manifest["identity_hash"]:
        raise GenerationError("shared_ledger_identity_mismatch")
    global_attempts = {entry["attempt_id"]: entry for entry in ledger["entries"]}
    if len(global_attempts) != len(ledger["entries"]):
        raise GenerationError("duplicate_shared_campaign_attempt_id")
    if set(local_attempts) - set(global_attempts):
        raise GenerationError("local_attempt_missing_from_shared_ledger; partial restore or ledger rollback")
    for attempt_id, entry in global_attempts.items():
        local = local_attempts.get(attempt_id)
        if local is None:
            # The global reservation is written first. A crash in this window
            # retains its charge and will halt in SharedBudget.locked().
            if entry["status"] != "reserved":
                raise GenerationError("shared_completed_attempt_missing_locally")
            continue
        provider, attempt = local
        if (entry["provider"] != provider or entry["input_id"] != attempt["input_id"]
            or entry["run_identity_hash"] != attempt["identity"]["run_identity_hash"]
            or entry["request_payload_hash"] != attempt["identity"]["request_payload_hash"]):
            raise GenerationError("shared_local_reservation_identity_mismatch")
        if entry["status"] != "reserved":
            for name in ("status", "failure_kind", "usage", "charged_input_tokens", "charged_output_tokens", "charged_cost_usd"):
                if entry[name] != attempt[name]:
                    raise GenerationError("shared_local_settlement_mismatch")
            if attempt["failure_kind"] in CRITICAL_FAILURES and not ledger["halt_reason"]:
                raise GenerationError("shared_critical_failure_halt_missing")
    return manifest, ledger


def totals(ledger: dict) -> dict:
    entries = ledger["entries"]
    return {
        "calls": len(entries), "retries": sum(entry["retry"] for entry in entries),
        "tokens": sum(entry["charged_input_tokens"] + entry["charged_output_tokens"] for entry in entries),
        "cost_usd_upper_bound": float(sum((Decimal(str(entry["charged_cost_usd"])) for entry in entries), Decimal(0))),
        "unknown_usage_attempts": sum(entry["usage"] is None for entry in entries),
    }


class SharedBudget:
    def __init__(self, directory):
        self.directory = writable(directory)
        self.manifest = None
        self.ledger = None

    @contextmanager
    def locked(self, *, run_identity: dict, campaign_id: str):
        """Hold the global campaign lock for an entire provider generation turn."""
        with (self.directory / "budget.lock").open("a+b") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GenerationError("shared_budget_locked") from None
            try:
                self.manifest, self.ledger = load_campaign(self.directory)
                identity = self.manifest["identity"]
                if campaign_id != identity["campaign_id"] or object_hash(run_identity) not in {
                    run["identity_hash"] for run in identity["runs"].values()
                }:
                    raise GenerationError("unapproved_run_identity")
                self._validate_ledger()
                if any(entry["status"] == "reserved" for entry in self.ledger["entries"]):
                    self.ledger["halt_reason"] = "unreconciled_prior_reservation"
                    self._save()
                if self.ledger["halt_reason"]:
                    raise BudgetStop("shared_budget_halted: " + self.ledger["halt_reason"])
                yield self
            finally:
                self.manifest = self.ledger = None
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _save(self):
        write_json(self.directory / "ledger.json", self.ledger)

    def _validate_ledger(self):
        targets = {(target["provider"], target["input_id"]): target for target in self.manifest["identity"]["targets"]}
        ids, prior_by_target = set(), {}
        for entry in self.ledger["entries"]:
            key = (entry["provider"], entry["input_id"])
            target = targets.get(key)
            if not target or entry["run_identity_hash"] != target["run_identity_hash"]:
                raise GenerationError("unapproved_shared_ledger_entry")
            if not re.fullmatch(r"[0-9a-f]{32}", entry["attempt_id"]) or entry["attempt_id"] in ids:
                raise GenerationError("invalid_shared_attempt_id")
            ids.add(entry["attempt_id"])
            prior = prior_by_target.setdefault(key, [])
            if entry["retry"] is not bool(prior) or any(item["status"] == "success" for item in prior):
                raise GenerationError("invalid_shared_retry_history")
            prior.append(entry)
            if entry["status"] not in ("reserved", "success", "failed"):
                raise GenerationError("invalid_shared_attempt_status")
            for name in ("charged_input_tokens", "charged_output_tokens"):
                if type(entry[name]) is not int or entry[name] < 0:
                    raise GenerationError("invalid_shared_token_charge")
            charge = entry["charged_cost_usd"]
            if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
                raise GenerationError("invalid_shared_cost_charge")
            from .providers import ProviderError, cost_ceiling, usage_upper_cost
            run = self.manifest["identity"]["runs"][entry["provider"]]
            config = read_json(self.directory / run["path"] / "config.json")
            pricing = config["pricing"]
            if entry["status"] == "reserved" or entry["usage"] is None:
                if (entry["charged_input_tokens"] < pricing["context_window_tokens"]
                    or entry["charged_output_tokens"] < config["max_output_tokens"]
                    or entry["charged_cost_usd"] < cost_ceiling(pricing["context_window_tokens"], config["max_output_tokens"], pricing)):
                    raise GenerationError("shared_unknown_usage_reservation_below_verified_ceiling")
            else:
                try:
                    inputs, outputs, upper_cost = usage_upper_cost(entry["provider"], entry["usage"], pricing)
                except ProviderError:
                    raise GenerationError("shared_known_usage_not_reconcilable") from None
                if entry["charged_input_tokens"] != inputs or entry["charged_output_tokens"] != outputs or entry["charged_cost_usd"] < upper_cost:
                    raise GenerationError("shared_charge_below_verified_usage")
            if entry["usage"] is None and entry["status"] != "reserved" and not self.ledger["halt_reason"]:
                raise GenerationError("shared_unknown_usage_halt_missing")
            if entry["status"] != "reserved":
                from .generation import _verify_artifact
                run = self.manifest["identity"]["runs"][entry["provider"]]
                run_directory = self.directory / run["path"]
                metadata_path = _verify_artifact(run_directory, entry["local_metadata_artifact"])
                metadata = read_json(metadata_path)
                for name in ("attempt_id", "input_id", "status", "usage", "charged_input_tokens", "charged_output_tokens", "charged_cost_usd"):
                    if metadata.get(name) != entry.get(name):
                        raise GenerationError("shared_local_attempt_mismatch")
                if metadata.get("identity", {}).get("run_identity_hash") != entry["run_identity_hash"]:
                    raise GenerationError("shared_local_identity_mismatch")
                for descriptor in metadata.get("artifacts", {}).values():
                    _verify_artifact(run_directory, descriptor)

    def reserve(self, *, run_identity_hash: str, provider: str, input_id: str, attempt_id: str,
                input_tokens: int, output_tokens: int, cost_usd: float, request_payload_hash: str) -> dict:
        if self.ledger is None:
            raise GenerationError("shared budget lock required")
        target = next((target for target in self.manifest["identity"]["targets"]
                       if (target["provider"], target["input_id"]) == (provider, input_id)), None)
        if not target or target["run_identity_hash"] != run_identity_hash:
            raise GenerationError("unapproved_shared_target")
        from .providers import cost_ceiling
        config = read_json(self.directory / self.manifest["identity"]["runs"][provider]["path"] / "config.json")
        pricing = config["pricing"]
        required_cost = cost_ceiling(pricing["context_window_tokens"], config["max_output_tokens"], pricing)
        if (type(input_tokens) is not int or input_tokens < pricing["context_window_tokens"]
            or type(output_tokens) is not int or output_tokens < config["max_output_tokens"]
            or type(cost_usd) not in (int, float) or not math.isfinite(cost_usd) or cost_usd < required_cost
            or not re.fullmatch(r"[0-9a-f]{32}", attempt_id)
            or not re.fullmatch(r"[0-9a-f]{64}", request_payload_hash)):
            raise GenerationError("shared_reservation_below_verified_ceiling")
        prior = [entry for entry in self.ledger["entries"] if (entry["provider"], entry["input_id"]) == (provider, input_id)]
        if any(entry["status"] == "success" for entry in prior):
            raise BudgetStop("successful_output_already_exists")
        budgets, used = self.manifest["identity"]["budgets"], totals(self.ledger)
        retry = bool(prior)
        if len(prior) >= budgets["max_attempts_per_input"]:
            raise BudgetStop("shared_input_attempt_limit")
        if used["calls"] + 1 > budgets["max_calls"]:
            raise BudgetStop("shared_call_limit")
        if used["retries"] + int(retry) > budgets["max_retries"]:
            raise BudgetStop("shared_retry_limit")
        if used["tokens"] + input_tokens + output_tokens > budgets["max_tokens"]:
            raise BudgetStop("shared_token_limit")
        charged = sum((Decimal(str(entry["charged_cost_usd"])) for entry in self.ledger["entries"]), Decimal(0))
        if charged + Decimal(str(cost_usd)) > Decimal(str(budgets["max_cost_usd"])):
            raise BudgetStop("shared_cost_limit")
        if output_tokens > budgets["max_output_tokens"]:
            raise BudgetStop("shared_output_limit")
        entry = {"attempt_id": attempt_id, "provider": provider, "input_id": input_id,
                 "run_identity_hash": run_identity_hash, "request_payload_hash": request_payload_hash,
                 "retry": retry, "status": "reserved", "reserved_at_utc": utc_now(),
                 "charged_input_tokens": input_tokens, "charged_output_tokens": output_tokens,
                 "charged_cost_usd": cost_usd, "usage": None, "failure_kind": None}
        self.ledger["entries"].append(entry)
        self._save()
        return entry

    def settle(self, attempt: dict) -> None:
        """Only release a reservation when usage is known and durably recorded."""
        if self.ledger is None:
            raise GenerationError("shared budget lock required")
        entry = next((entry for entry in self.ledger["entries"] if entry["attempt_id"] == attempt["attempt_id"]), None)
        if not entry or entry["status"] != "reserved":
            raise GenerationError("shared_reservation_missing_or_completed")
        if entry["run_identity_hash"] != attempt["identity"]["run_identity_hash"]:
            raise GenerationError("shared_settlement_identity_mismatch")
        entry["status"] = attempt["status"]
        entry["failure_kind"] = attempt["failure_kind"]
        entry["finished_at_utc"] = utc_now()
        entry["local_metadata_artifact"] = attempt["metadata_artifact"]
        if attempt["usage"] is None:
            self.ledger["halt_reason"] = "unknown_usage"
        else:
            from .providers import ProviderError, usage_upper_cost
            config = read_json(self.directory / self.manifest["identity"]["runs"][entry["provider"]]["path"] / "config.json")
            try:
                inputs, outputs, cost = usage_upper_cost(entry["provider"], attempt["usage"], config["pricing"])
            except ProviderError:
                self.ledger["halt_reason"] = "billing_reconciliation_failure"
                self._save()
                raise GenerationError("shared_known_usage_not_reconcilable") from None
            if attempt["charged_input_tokens"] != inputs or attempt["charged_output_tokens"] != outputs or attempt["charged_cost_usd"] < cost:
                self.ledger["halt_reason"] = "billing_reconciliation_failure"
                self._save()
                raise GenerationError("shared_charge_below_verified_usage")
            if any(attempt[name] > entry[name] for name in ("charged_input_tokens", "charged_output_tokens", "charged_cost_usd")):
                self.ledger["halt_reason"] = "usage_exceeded_verified_reservation"
            for name in ("charged_input_tokens", "charged_output_tokens", "charged_cost_usd", "usage"):
                entry[name] = attempt[name]
            if attempt["failure_kind"] in CRITICAL_FAILURES:
                self.ledger["halt_reason"] = attempt["failure_kind"]
        self._save()


def campaign_summary(directory) -> dict:
    manifest, ledger = load_campaign(directory)
    validator = SharedBudget(directory)
    validator.manifest, validator.ledger = manifest, ledger
    validator._validate_ledger()
    return {"campaign_id": manifest["identity"]["campaign_id"], "identity_hash": manifest["identity_hash"],
            "budgets": manifest["identity"]["budgets"], "used": totals(ledger),
            "halt_reason": ledger["halt_reason"], "api_status": "NOT_RUN" if not ledger["entries"] else (
                "FIXTURE_ONLY" if manifest["identity"]["provenance"] == "fixture_only" else "ATTEMPTED")}
