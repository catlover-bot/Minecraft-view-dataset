"""Read-only parent validation and disabled, identity-bound resmoke preparation.

The immutable old campaign remains readable by its original validators. Only
new files are written. Input bodies and generation metadata are kept separate:
ordinary requests contain their original description, never a GT descriptor.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from offline_evaluation.server_v2.core import Limits, grid_to_payload
from offline_evaluation.server_v2.data import load_scene, sha256_file
from offline_evaluation.server_v2.generation import (
    GenerationConfig, GenerationError, _load_run, digest, object_hash,
    read_json, utc_now, write_json,
)
from offline_evaluation.server_v2.io_utils import ROOT, writable
from offline_evaluation.server_v2.shared_budget import campaign_summary

PROTOCOL_ID = "goal004_native_cuboids_best_effort_fixed16_v1"
MODELS = {"openai": "gpt-5-mini", "anthropic": "claude-haiku-4-5-20251001"}
SCENES = ("buildings_100_v4_building_060", "buildings_100_v4_building_075")
PARENT_DIRECTORY = Path("/root/minecraft-runs/goal002-approved-smoke-001")
PARENT_CAMPAIGN_ID = "665b1c3eb41a45e3b5de95f6143f7eaa"
PARENT_USAGE = {"calls": 14, "retries": 4, "tokens": 173509,
                "cost_usd_upper_bound": 0.25857425, "unknown_usage_attempts": 0}
AUTHORITY_PATH = Path(f"/root/minecraft-runs/.resmoke-v2-{PARENT_CAMPAIGN_ID}.json")
PROMPT_PATH = Path(__file__).parent / "prompts/builder.txt"
PRICING_PATH = ROOT / "reports/server_only/goal004/official_pricing.json"
CAP_REPORT_PATH = ROOT / "reports/server_only/goal004/cuboid_cpu.json"
OUTPUT_CAP = 32768


def safe_path(path) -> Path:
    candidate = Path(path).absolute()
    if any(p.is_symlink() for p in (candidate, *candidate.parents)):
        raise GenerationError("symlink path forbidden")
    return writable(candidate)


def implementation_hashes() -> dict:
    """Bind new execution plus the exact reused pure/IO/analysis components."""
    directory = Path(__file__).parent
    own = [p for p in directory.rglob("*.py") if "tests" not in p.parts]
    shared = [ROOT / "offline_evaluation/server_v2" / name for name in (
        "generation.py", "providers.py", "core.py", "alignment.py", "data.py",
        "io_utils.py", "shared_budget.py", "smoke.py", "oracle.py", "geometry_data.py",
        "cohort_v4.py", "analysis_v4.py", "statistics.py", "checkpoint.py",
    )]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(own + shared)}


def parent_snapshot(parent_directory=PARENT_DIRECTORY) -> dict:
    directory = safe_path(parent_directory)
    summary = campaign_summary(directory)  # Includes old run and ledger validation.
    return {"campaign_directory": str(directory), "campaign_id": summary["campaign_id"],
            "identity_hash": summary["identity_hash"], "usage": summary["used"],
            "campaign_sha256": sha256_file(directory / "campaign.json"),
            "approval_sha256": sha256_file(directory / "approval.json"),
            "ledger_sha256": sha256_file(directory / "ledger.json"),
            "state_sha256": {p: sha256_file(directory / "runs" / p / "state.json") for p in MODELS}}


def verify_parent(parent: dict) -> None:
    if parent_snapshot(parent["campaign_directory"]) != parent:
        raise GenerationError("parent_changed; replan required; never reset or rebind the prior ledger")


def verify_gt_sources(old_records: list[dict], descriptors: dict) -> None:
    for record in old_records:
        if record["condition"] == "E":
            descriptor = descriptors[record["scene_id"]]
            if (descriptor["voxels_sha256"] != record["source_sha256"]
                or descriptor["bbox_sha256"] != record["source_GT_bbox_sha256"]):
                raise GenerationError("GT differs from the frozen old smoke target")


def budgets(pricing: dict) -> dict:
    maximum = sum(8 * (pricing[p]["context_window_tokens"] + OUTPUT_CAP) for p in MODELS)
    return {"max_new_calls": 16, "max_total_calls": 30, "max_new_retries": 0,
            "max_total_retries": 4, "max_attempts_per_input": 1, "concurrency": 1,
            "max_cost_usd": 5.0, "max_tokens": maximum,
            "max_total_tokens": PARENT_USAGE["tokens"] + maximum, "max_output_tokens": OUTPUT_CAP}


def validate_pricing(pricing: dict, *, fresh: bool = False) -> None:
    if set(pricing) != set(MODELS):
        raise GenerationError("both fixed model pricing records required")
    for provider, model in MODELS.items():
        record = pricing[provider]
        if (record.get("provider") != provider or record.get("model") != model
                or record.get("verified") is not True
                or record.get("model_max_output_tokens", 0) < OUTPUT_CAP):
            raise GenerationError("invalid fixed model pricing")
        if fresh:
            # Same UTC-day official verification required before a paid start.
            if record.get("verified_on") != utc_now()[:10]:
                raise GenerationError("pricing_documentation_requires_today_verification_and_replan")
            GenerationConfig(provider=provider, model=model, api_allowed=True,
                max_calls=16, max_tokens=budgets(pricing)["max_tokens"], max_cost_usd=5,
                max_output_tokens=OUTPUT_CAP, max_attempts_per_input=1,
                pricing=record, pricing_max_age_days=1).require_generation(True)


def make_inputs(old_records: list[dict], grids: dict) -> list[dict]:
    """GT is accessed only in E; P retains the old attributes-only body exactly."""
    from .cuboids import encode_exact, exact_builder_text
    from .schema import schema_sha256 as schema_hash
    prompt_hash = digest(PROMPT_PATH.read_bytes())
    indexed = {(r["scene_id"], r["condition"]): r for r in old_records}
    if set(indexed) != {(s, c) for s in SCENES for c in "OJPE"} or len(old_records) != 8:
        raise GenerationError("old fixed smoke input set mismatch")
    records = []
    for scene in SCENES:
        for condition in "OJPE":
            old = indexed[scene, condition]
            text = old["input_text"]
            if digest(text.encode()) != old["input_sha256"]:
                raise GenerationError("old body hash mismatch")
            if condition == "E":
                text = exact_builder_text(encode_exact(grids[scene]))
            record = {"scene_id": scene, "condition": condition,
                "condition_label": {"O": "Original saved description", "J": "Lossless JSON reserialization",
                    "P": "GT-attributes-only diagnostic", "E": "Complete discrete cuboid specification diagnostic"}[condition],
                "description_model": "openai" if condition in "OJ" else None,
                "input_text": text, "input_sha256": digest(text.encode()),
                "source_input_id": old["input_id"], "source_input_sha256": old["input_sha256"],
                "source_sha256": old["source_sha256"], "protocol_id": PROTOCOL_ID,
                "prompt_sha256": prompt_hash, "schema_sha256": schema_hash(),
                "provenance": "new_implementation"}
            # Same semantic input ID across providers; schedule adds provider.
            record["input_id"] = object_hash(record)
            records.append(record)
    return records


def prepare(destination, *, pricing_file, commit: str) -> dict:
    """Create a unique disabled plan. Never creates approval or enabled state."""
    from .schema import action_schema, schema_sha256 as schema_hash
    destination = safe_path(destination)
    if destination.exists():
        raise GenerationError("new unique preparation directory required")
    if not destination.is_relative_to(Path("/root/minecraft-runs")):
        raise GenerationError("production preparation must be in /root/minecraft-runs")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise GenerationError("explicit forty-digit code commit required")
    if AUTHORITY_PATH.exists():
        raise GenerationError("parent resmoke already activated; resume its original plan")
    pricing_record = read_json(Path(pricing_file))
    if Path(pricing_file).resolve() != PRICING_PATH or sha256_file(Path(pricing_file)) != sha256_file(PRICING_PATH):
        raise GenerationError("use the newly verified Goal004 official pricing record")
    pricing = {provider: pricing_record[provider] for provider in MODELS}
    validate_pricing(pricing, fresh=True)
    parent = parent_snapshot()
    if parent["campaign_id"] != PARENT_CAMPAIGN_ID or parent["usage"] != PARENT_USAGE:
        raise GenerationError("parent usage differs from Goal004 proposal; replan required")
    _, _, old_records, _, _ = _load_run(PARENT_DIRECTORY / "runs/openai")
    _, _, second_records, _, _ = _load_run(PARENT_DIRECTORY / "runs/anthropic")
    if old_records != second_records:
        raise GenerationError("old provider input sets disagree")
    grids, gt = {}, {}
    for scene in SCENES:
        building = ROOT / "datasets/buildings_100_v4" / scene.removeprefix("buildings_100_v4_")
        loaded = load_scene(building)
        grids[scene] = loaded.grid
        gt[scene] = {"building_directory": str(building), "voxels_sha256": sha256_file(building / "gt/voxels.npy"),
            "bbox_sha256": sha256_file(building / "gt/bbox.json"), "grid_sha256": object_hash(grid_to_payload(loaded.grid))}
    verify_gt_sources(old_records, gt)
    records = make_inputs(old_records, grids)
    schedule = [{**record, "provider": provider, "model": model, "schedule_index": 2 * index + offset}
                for index, record in enumerate(records) for offset, (provider, model) in enumerate(MODELS.items())]
    identity = {"protocol_id": PROTOCOL_ID, "provenance": "new_implementation", "commit": commit,
        "plan_path": str(destination / "plan.json"), "enabled_directory": str(destination / "enabled"),
        "authority_path": str(AUTHORITY_PATH), "parent": parent, "schedule": schedule,
        "prompt": PROMPT_PATH.read_text(), "prompt_sha256": digest(PROMPT_PATH.read_bytes()),
        "schema": action_schema(), "schema_sha256": schema_hash(), "pricing": pricing,
        "pricing_record_sha256": sha256_file(Path(pricing_file)), "budgets": budgets(pricing),
        "output_cap_cpu_evidence_sha256": sha256_file(CAP_REPORT_PATH), "timeout_seconds": 600.0,
        "limits": dataclasses.asdict(Limits()), "implementation_hashes": implementation_hashes(),
        "gt_by_scene": gt, "scope": "fixed16_only_no_retries_no_full960",
        "reservation_policy": "full documented context input plus output cap; includes schema framing within context; no byte-to-token certification",
        "account_structured_output_acceptance": "NOT_RUN"}
    plan = {"identity": identity, "identity_hash": object_hash(identity), "created_at_utc": utc_now(),
            "api_allowed": False, "research_api_additional_sends": 0,
            "status": "PROTOCOL_READY_AWAITING_EXPLICIT_RESMOKE_APPROVAL"}
    # All validation above precedes writing. The target has never existed.
    destination.mkdir(parents=True)
    write_json(destination / "plan.json", plan)
    return load_plan(destination / "plan.json")


def load_plan(path) -> dict:
    """Reject relocated, edited, stale-code, partial-provider or expanded plans."""
    from .schema import action_schema, schema_sha256 as schema_hash
    path = safe_path(path)
    if path.is_dir():
        path = path / "plan.json"
    if not path.is_relative_to(Path("/root/minecraft-runs")):
        raise GenerationError("canonical production plan must remain in /root/minecraft-runs")
    plan = read_json(path)
    identity = plan["identity"]
    if plan.get("api_allowed") is not False or plan.get("identity_hash") != object_hash(identity):
        raise GenerationError("disabled plan identity mismatch")
    if identity.get("plan_path") != str(path):
        raise GenerationError("canonical plan path mismatch; copied plans cannot activate")
    if (identity.get("protocol_id") != PROTOCOL_ID or identity.get("provenance") != "new_implementation"
        or identity.get("scope") != "fixed16_only_no_retries_no_full960"):
        raise GenerationError("unsupported protocol scope")
    if (identity.get("enabled_directory") != str(path.parent / "enabled")
        or identity.get("authority_path") != str(AUTHORITY_PATH)
        or identity.get("implementation_hashes") != implementation_hashes()):
        raise GenerationError("canonical destination or implementation mismatch")
    if (identity.get("prompt") != PROMPT_PATH.read_text()
        or identity.get("prompt_sha256") != digest(PROMPT_PATH.read_bytes())
        or identity.get("schema") != action_schema() or identity.get("schema_sha256") != schema_hash()):
        raise GenerationError("prompt or schema mismatch")
    validate_pricing(identity["pricing"])
    if (identity.get("timeout_seconds") != 600.0
        or identity.get("output_cap_cpu_evidence_sha256") != sha256_file(CAP_REPORT_PATH)
        or read_json(CAP_REPORT_PATH)["output_cap_assessment"]["proposed_tokens"] != OUTPUT_CAP):
        raise GenerationError("output cap evidence or request timeout mismatch")
    pricing_record = read_json(PRICING_PATH)
    if (identity["pricing_record_sha256"] != sha256_file(PRICING_PATH)
        or identity["pricing"] != {p: pricing_record[p] for p in MODELS}):
        raise GenerationError("frozen official pricing record mismatch")
    if identity["budgets"] != budgets(identity["pricing"]):
        raise GenerationError("budget scope mismatch")
    parent = identity["parent"]
    if (parent["campaign_directory"] != str(PARENT_DIRECTORY) or parent["campaign_id"] != PARENT_CAMPAIGN_ID
        or parent["usage"] != PARENT_USAGE):
        raise GenerationError("parent scope mismatch")
    schedule = identity["schedule"]
    expected = [(s, c, p) for s in SCENES for c in "OJPE" for p in MODELS]
    if [(r["scene_id"], r["condition"], r["provider"]) for r in schedule] != expected:
        raise GenerationError("fixed interleaved sixteen schedule required")
    for index, record in enumerate(schedule):
        if (record["model"] != MODELS[record["provider"]] or record["schedule_index"] != index
            or digest(record["input_text"].encode()) != record["input_sha256"]
            or record["schema_sha256"] != schema_hash() or record["prompt_sha256"] != identity["prompt_sha256"]):
            raise GenerationError("input identity mismatch")
        core = {k: v for k, v in record.items() if k not in {"input_id", "provider", "model", "schedule_index"}}
        if object_hash(core) != record["input_id"]:
            raise GenerationError("input ID mismatch")
        if record["condition"] in "OJP" and record["input_sha256"] != record["source_input_sha256"]:
            raise GenerationError("original O/J/P body was changed")
    for first, second in zip(schedule[::2], schedule[1::2]):
        if first["input_id"] != second["input_id"]:
            raise GenerationError("providers must receive identical input bodies")
    if identity["limits"] != dataclasses.asdict(Limits()):
        raise GenerationError("shared executor limit mismatch")
    if set(identity["gt_by_scene"]) != set(SCENES):
        raise GenerationError("GT scene scope mismatch")
    for scene, descriptor in identity["gt_by_scene"].items():
        directory = ROOT / "datasets/buildings_100_v4" / scene.removeprefix("buildings_100_v4_")
        if descriptor["building_directory"] != str(directory):
            raise GenerationError("GT source directory mismatch")
        if (sha256_file(directory / "gt/voxels.npy") != descriptor["voxels_sha256"]
            or sha256_file(directory / "gt/bbox.json") != descriptor["bbox_sha256"]):
            raise GenerationError("GT source hash mismatch")
    # A self-consistent rehash cannot substitute different saved descriptions or
    # a partial E. Compare with the immutable parent's bodies and exact GT anew.
    _, _, original_records, _, _ = _load_run(PARENT_DIRECTORY / "runs/openai")
    verify_gt_sources(original_records, identity["gt_by_scene"])
    grids = {scene: load_scene(Path(value["building_directory"])).grid
             for scene, value in identity["gt_by_scene"].items()}
    expected_records = make_inputs(original_records, grids)
    expected_schedule = [{**record, "provider": provider, "model": model, "schedule_index": 2 * index + offset}
                for index, record in enumerate(expected_records) for offset, (provider, model) in enumerate(MODELS.items())]
    if schedule != expected_schedule:
        raise GenerationError("schedule differs from fixed parent descriptions or exact GT")
    return plan
