"""Manual approval launcher for the fixed16-input, two-provider smoke only.

Preparation and dry-run cannot send requests. Approval creates new enabled
directories, preserving disabled Builder preparations. Each proposal has one
activation, and its ledger is bound to that canonical directory. Nothing here
advances to the planned960-input evaluation experiment.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import uuid

from . import providers
from .generation import (GenerationConfig, GenerationError, _implementation_hashes, _load_run,
                         digest, generate, object_hash, prepare_generation, read_json, run_lock,
                         utc_now, write_json)
from .io_utils import writable
from .shared_budget import campaign_summary, validate_budgets


MODELS = {"openai": "gpt-5-mini", "anthropic": "claude-haiku-4-5-20251001"}


def _enabled_config(config: GenerationConfig, campaign_id: str, budgets: dict) -> GenerationConfig:
    return dataclasses.replace(config, api_allowed=True, budget_campaign_id=campaign_id,
                               max_calls=budgets["max_calls"], max_tokens=budgets["max_tokens"],
                               max_cost_usd=budgets["max_cost_usd"], max_output_tokens=budgets["max_output_tokens"],
                               max_attempts_per_input=budgets["max_attempts_per_input"])


def _proposal(directory) -> dict:
    directory = writable(directory)
    proposal = read_json(directory / "proposal.json")
    identity = proposal["identity"]
    if proposal.get("identity_hash") != object_hash(identity) or identity["implementation_hashes"] != _implementation_hashes():
        raise GenerationError("smoke_proposal_identity_mismatch")
    validate_budgets(identity["budgets"])
    for provider, run in identity["runs"].items():
        with run_lock(directory / "runs" / provider):
            manifest, _, _, _, state = _load_run(directory / "runs" / provider)
        if object_hash(manifest["identity"]) != run["disabled_identity_hash"] or state["attempts"]:
            raise GenerationError("smoke_disabled_preparation_changed")
    return proposal


def prepare_smoke_campaign(disabled_runs: dict, campaign_dir, *, max_tokens: int | None = None) -> dict:
    """Freeze a disabled campaign and the exact future approval scope; no keys read."""
    if set(disabled_runs) != {"openai", "anthropic"}:
        raise GenerationError("both fixed smoke providers required")
    directory = writable(campaign_dir)
    if directory.exists():
        raise GenerationError("new unique smoke proposal directory required")
    loaded = {}
    for provider in ("openai", "anthropic"):
        source = writable(disabled_runs[provider])
        with run_lock(source):
            manifest, config, records, prompt, state = _load_run(source)
        if config.api_allowed or config.budget_campaign_id or state["attempts"]:
            raise GenerationError("smoke source must be disabled and unattempted")
        if config.provider != provider or (config.provenance != "fixture_only" and config.model != MODELS[provider]):
            raise GenerationError("fixed smoke provider/model mismatch; no fallback")
        if len(records) != 8 or any(record.get("status", "prepared") != "prepared" for record in records):
            raise GenerationError("exactly8 feasible inputs per provider required")
        keys = {(record["scene_id"], record["condition"]) for record in records}
        scenes = {record["scene_id"] for record in records}
        if len(scenes) != 2 or keys != {(scene, condition) for scene in scenes for condition in "OJPE"}:
            raise GenerationError("smoke requires2 fixed buildings and O/J/P/E each once")
        if any(record.get("description_model") != "openai" for record in records if record["condition"] in "OJ"):
            raise GenerationError("smoke O/J must use saved OpenAI descriptions")
        if not isinstance(config.pricing, dict) or type(config.pricing.get("model_max_output_tokens")) is not int or config.pricing["model_max_output_tokens"] < 16_384:
            raise GenerationError("verified model output ceiling16384 required")
        loaded[provider] = manifest, config, records, prompt
    if {record["input_id"] for record in loaded["openai"][2]} != {record["input_id"] for record in loaded["anthropic"][2]}:
        raise GenerationError("providers must use the same fixed smoke inputs")
    if loaded["openai"][2] != loaded["anthropic"][2] or loaded["openai"][3] != loaded["anthropic"][3]:
        raise GenerationError("providers must share complete inputs and prompt")
    if loaded["openai"][1].limits != loaded["anthropic"][1].limits:
        raise GenerationError("providers must share validator/executor limits")
    maximum_context = max(item[1].pricing["context_window_tokens"] for item in loaded.values())
    budgets = validate_budgets({"max_calls": 20, "max_retries": 4, "max_attempts_per_input": 2,
                               "max_cost_usd": 5, "max_tokens": max_tokens if max_tokens is not None else 20 * (maximum_context + 16_384),
                               "max_output_tokens": 16_384, "concurrency": 1})
    campaign_id = uuid.uuid4().hex
    planned = {}
    for provider, (manifest, config, records, prompt) in loaded.items():
        enabled = _enabled_config(config, campaign_id, budgets)
        enabled.require_generation(True)
        planned[provider] = enabled
    # All checks finish before writing the new proposal.
    directory.mkdir(parents=True)
    with run_lock(directory):
        runs, targets = {}, []
        for provider, (source_manifest, config, records, prompt) in loaded.items():
            disabled = dataclasses.replace(planned[provider], api_allowed=False)
            manifest = prepare_generation(records, disabled, directory / "runs" / provider,
                                          commit=source_manifest["identity"]["commit"], prompt=prompt)
            enabled_identity = {**manifest["identity"], "config_hash": object_hash(planned[provider].to_dict())}
            runs[provider] = {"disabled_identity_hash": object_hash(manifest["identity"]),
                              "enabled_identity_hash": object_hash(enabled_identity),
                              "enabled_config_hash": object_hash(planned[provider].to_dict()),
                              "source_run_identity_hash": object_hash(source_manifest["identity"]),
                              "source_inputs_hash": source_manifest["identity"]["inputs_hash"],
                              "prompt_hash": manifest["identity"]["prompt_hash"], "model": config.model,
                              "commit": source_manifest["identity"]["commit"]}
            for record in records:
                targets.append({"provider": provider, "input_id": record["input_id"],
                                "scene_id": record["scene_id"], "condition": record["condition"],
                                "input_hash": object_hash(record), "source_sha256": record.get("source_sha256"),
                                "run_identity_hash": object_hash(enabled_identity)})
        identity = {"version": 1, "purpose": "goal002_v4_two_builder_smoke", "campaign_id": campaign_id,
                    "provenance": loaded["openai"][1].provenance, "implementation_hashes": _implementation_hashes(),
                    "budgets": budgets, "runs": runs, "targets": targets}
        proposal = {"identity": identity, "identity_hash": object_hash(identity), "created_at_utc": utc_now(),
                    "api_allowed": False, "api_status": "NOT_RUN", "account_model_availability": "NOT_RUN"}
        write_json(directory / "proposal.json", proposal)
        write_json(directory / "approval_template.json", _approval_template(proposal))
        return proposal


def _approval_template(proposal: dict) -> dict:
    identity = proposal["identity"]
    return {"manual_confirmed": False, "proposal_identity_hash": proposal["identity_hash"],
            "campaign_id": identity["campaign_id"], "target_inputs": identity["targets"],
            "budgets": identity["budgets"], "models": {provider: run["model"] for provider, run in identity["runs"].items()},
            "enabled_config_hashes": {provider: run["enabled_config_hash"] for provider, run in identity["runs"].items()},
            "source_run_identity_hashes": {provider: run["source_run_identity_hash"] for provider, run in identity["runs"].items()},
            "prompt_hashes": {provider: run["prompt_hash"] for provider, run in identity["runs"].items()},
            "implementation_hashes": identity["implementation_hashes"],
            "enabled_directory": None,
            "scope": "only fixed16 smoke inputs; no automatic full experiment"}


def approval_template(campaign_dir, *, enabled_directory=None) -> dict:
    template = _approval_template(_proposal(campaign_dir))
    if enabled_directory is not None:
        template["enabled_directory"] = str(writable(enabled_directory))
    return template


def approve_smoke_campaign(proposal_dir, new_enabled_dir, approval: dict) -> dict:
    """Create enabled artifacts only after explicit matching manual confirmation.

    This function does not send requests. Do not invoke it as approval during a
    preparation-only Goal; a user must separately approve these exact targets,
    provider models, cost/call/token/retry caps and source identities first.
    """
    proposal_dir, directory = writable(proposal_dir), writable(new_enabled_dir)
    with run_lock(proposal_dir):
        proposal = _proposal(proposal_dir)
        expected = {**_approval_template(proposal), "manual_confirmed": True, "enabled_directory": str(directory)}
        if approval != expected:
            raise GenerationError("explicit matching manual smoke approval required")
        if directory.exists() or directory == proposal_dir:
            raise GenerationError("new unique enabled smoke directory required")
        if (proposal_dir / "activation.json").exists():
            raise GenerationError("proposal already activated; resume its existing ledger")
        # Reserve this activation before creating usable enabled artifacts. A
        # partial activation cannot be reissued in another directory.
        write_json(proposal_dir / "activation.json", {"status": "reserved", "enabled_directory": str(directory),
                                                     "approval_hash": object_hash(approval), "created_at_utc": utc_now()})
        directory.mkdir(parents=True)
        identity = proposal["identity"]
        runs = {}
        for provider, planned in identity["runs"].items():
            source = proposal_dir / "runs" / provider
            with run_lock(source):
                manifest, config, records, prompt, _ = _load_run(source)
            enabled = _enabled_config(config, identity["campaign_id"], identity["budgets"])
            enabled.require_generation(True)
            if object_hash(enabled.to_dict()) != planned["enabled_config_hash"]:
                raise GenerationError("enabled config approval mismatch")
            run_manifest = prepare_generation(records, enabled, directory / "runs" / provider,
                                              commit=planned["commit"], prompt=prompt)
            if object_hash(run_manifest["identity"]) != planned["enabled_identity_hash"]:
                raise GenerationError("enabled run approval mismatch")
            runs[provider] = {"path": f"runs/{provider}", "identity_hash": planned["enabled_identity_hash"]}
        enabled_identity = {"version": 1, "purpose": identity["purpose"], "campaign_id": identity["campaign_id"],
                            "provenance": identity["provenance"], "implementation_hashes": identity["implementation_hashes"],
                            "budgets": identity["budgets"], "targets": identity["targets"], "runs": runs,
                            "proposal_identity_hash": proposal["identity_hash"], "approval_hash": object_hash(approval),
                            "ledger_directory": str(directory)}
        campaign = {"identity": enabled_identity, "identity_hash": object_hash(enabled_identity),
                    "api_allowed": True, "created_at_utc": utc_now(), "account_model_availability": "NOT_RUN"}
        write_json(directory / "approval.json", approval)
        write_json(directory / "ledger.json", {"identity_hash": campaign["identity_hash"], "entries": [], "halt_reason": None})
        write_json(directory / "campaign.json", campaign)
        write_json(proposal_dir / "activation.json", {"status": "complete", "enabled_directory": str(directory),
                                                     "approval_hash": object_hash(approval), "campaign_identity_hash": campaign["identity_hash"]})
        return campaign


def dry_run_smoke_campaign(campaign_dir) -> dict:
    directory = writable(campaign_dir)
    if (directory / "proposal.json").exists():
        proposal = _proposal(directory)
        result = {"campaign_id": proposal["identity"]["campaign_id"], "identity_hash": proposal["identity_hash"],
                  "api_allowed": False, "api_status": "NOT_RUN", "status": "SMOKE_READY_AWAITING_BUDGET_APPROVAL",
                  "budgets": proposal["identity"]["budgets"], "target_count": len(proposal["identity"]["targets"])}
    else:
        result = {**campaign_summary(directory), "api_allowed": True, "target_count": 16}
    result["api_key_present"] = {provider: bool(os.environ.get(name)) for provider, name in providers.KEY_NAMES.items()}
    result["account_model_availability"] = "NOT_RUN"
    return result


def run_smoke_campaign(campaign_dir, *, permission: bool = False, transports: dict | None = None) -> dict:
    """Explicit future paid invocation; sequential and limited to the approved16."""
    from .shared_budget import load_campaign
    directory = writable(campaign_dir)
    if permission is not True:
        raise GenerationError("explicit paid smoke invocation permission required")
    if (directory / "proposal.json").exists():
        raise GenerationError("campaign_api_disabled")
    manifest, _ = load_campaign(directory)
    if transports is not None and manifest["identity"]["provenance"] != "fixture_only":
        raise GenerationError("mock transports require fixture_only unit tests")
    if not all(bool(os.environ.get(name)) for name in providers.KEY_NAMES.values()):
        raise GenerationError("missing_api_key; both provider keys required before any smoke request")
    results = {}
    for provider in ("openai", "anthropic"):
        results[provider] = generate(directory / manifest["identity"]["runs"][provider]["path"], permission=True,
                                     transport=(transports or {}).get(provider), campaign_dir=directory)
        if campaign_summary(directory)["halt_reason"]:
            break
    return {**campaign_summary(directory), "providers": results, "full_experiment": "NOT_RUN"}
