"""Immutable scope, canonical path, exact wire, and source binding for Goal015C."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

from . import adapter, pricing

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = Path("/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03")
REGISTRY = ROOT / "reports/server_only/goal015c/EXECUTION_BINDING.json"
PARENT = Path("/root/minecraft-runs/goal015-20261017T170546Z")
CHILD = Path("/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79")
A, B, C = adapter.CONDITIONS
EXPECTED_SCHEDULE = [
    ("openai", A, 1), ("anthropic", C, 1), ("openai", B, 1),
    ("anthropic", A, 1), ("openai", C, 1), ("anthropic", B, 1),
    ("anthropic", B, 2), ("openai", C, 2), ("anthropic", A, 2),
    ("openai", B, 2), ("anthropic", C, 2), ("openai", A, 2),
]


class PlanError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise PlanError(message)


def path_safe(value):
    p = Path(value)
    require(p.is_absolute() and ".." not in p.parts, "absolute canonical path required")
    require(not any(x.is_symlink() for x in (p, *p.parents)), "symlink path refused")
    return p


def descriptor(path):
    return pricing.descriptor(path_safe(Path(path).absolute()))


def plan_identity(plan):
    value = {k: v for k, v in plan.items() if k != "plan_id"}
    return hashlib.sha256(adapter.canonical_json(value).encode()).hexdigest()


def read_json(path):
    from offline_evaluation.server_v2.core import strict_json, Limits
    return strict_json(Path(path).read_text(), Limits(max_json_bytes=32_000_000))


def check_file(value):
    require(descriptor(value["path"]) == value, "bound file changed")


def _verify_sources(plan):
    bindings = plan["source_bindings"]
    expected_keys = {"parent_plan", "parent_ledger", "parent_approval", "parent_authority",
                     "parent_execution_lock", "child_plan", "child_required_absent"}
    require(set(bindings) == expected_keys, "source binding scope differs")
    for name in expected_keys - {"child_required_absent"}:
        check_file(bindings[name])
    for name in bindings["child_required_absent"]:
        require(not os.path.lexists(path_safe(name)), "old child state appeared")
    if plan.get("fixture_only") is True:
        return
    expected = {"parent_plan": PARENT/"plan.json", "parent_ledger": PARENT/"enabled/ledger.json",
                "parent_approval": PARENT/"enabled/approval.json", "parent_authority": PARENT/"enabled/authority.json",
                "parent_execution_lock": PARENT/"execution.lock", "child_plan": CHILD/"plan.json"}
    require(all(bindings[k]["path"] == str(p) for k, p in expected.items()), "wrong old run binding")
    require(bindings["child_required_absent"] == [str(CHILD / n) for n in
            ("enabled", "execution.lock", "completion.lock")], "child absence scope differs")
    entry = read_json(CANONICAL_ROOT/"ENTRY.json")
    for k, p in expected.items():
        require(bindings[k] == entry["protected_files"][str(p)], "old run differs from entry snapshot")
    ledger = read_json(expected["parent_ledger"])
    states = [r["state"] for r in ledger["calls"]]
    require(states.count("SETTLED") == 72 and states.count("NOT_SENT") == 8
            and set(states) == {"SETTLED", "NOT_SENT"}, "parent settlement changed")
    known = sum((Decimal(r.get("known_cost_usd") or "0") for r in ledger["calls"]), Decimal(0))
    require(known == Decimal("4.43765700"), "parent cost changed")
    child = read_json(expected["child_plan"])
    require(child["api_allowed"] is False and child["budget_approved"] is False, "old child is enabled")


def verify_plan(plan_path):
    path = path_safe(Path(plan_path).absolute())
    plan = read_json(path)
    require(plan.get("schema") == "goal015c_plan_v1" and path.name == "plan.json", "wrong diagnostic plan")
    require(plan.get("plan_id") == plan_identity(plan), "plan identity mismatch")
    root = path_safe(plan["canonical_root"])
    require(path == root/"plan.json", "copied plan refused")
    fixture = plan.get("fixture_only") is True
    if fixture:
        require(root != CANONICAL_ROOT and not (root.parent == Path("/root/minecraft-runs")
                and root.name.startswith("goal015c-")), "production root cannot be fixture")
    else:
        require(root == CANONICAL_ROOT, "unregistered diagnostic root")
        registry = read_json(REGISTRY)
        require(registry["plan"] == descriptor(path) and registry["plan_id"] == plan["plan_id"]
                and registry["canonical_root"] == str(root), "registered plan binding mismatch")
    require(plan["api_allowed"] is False and plan["budget_approved"] is False,
            "plan flags must stay disabled; authority is separate")
    require(plan["scope_calls"] == 12 and plan["concurrency"] == 1 and plan["retries"] == 0
            and plan["timeout_seconds"] == 600, "diagnostic scope changed")
    require(plan["pricing_contract"] == pricing.contract(), "diagnostic contract changed")
    require(plan["task_summary"] == adapter.TASK_SUMMARY
            and adapter.canonical_json(plan["expected_document"]) == adapter.canonical_json(adapter.EXPECTED_DOCUMENT),
            "mini-task changed")
    require(Decimal(plan["proposed_budget_usd"]) > 0 and Decimal(plan["proposed_budget_usd"]) <= Decimal("3.25"),
            "invalid proposed diagnostic budget")
    if not fixture:
        require(plan["proposed_budget_usd"] == "3.25", "production cap changed")
        require(plan.get("implementation_hashes") and plan.get("specification"), "implementation/spec binding absent")
    for name, expected in plan.get("implementation_hashes", {}).items():
        require(descriptor(name)["sha256"] == expected, "implementation dependency changed")
    if plan.get("specification"):
        check_file(plan["specification"])
    for name in ("prompt", "task_input", "launcher", "local_order_audit"):
        if name in plan:
            check_file(plan[name])
    require(len(plan["schedule"]) == 12, "exactly twelve calls required")
    reserve_total = Decimal(0)
    for i, (row, expected) in enumerate(zip(plan["schedule"], EXPECTED_SCHEDULE, strict=True), 1):
        builder, condition, repeat = expected
        require(row["call_id"] == f"goal015c_call_{i:03d}"
                and (row["builder"], row["condition"], row["repeat_index"]) == expected,
                "fixed diagnostic schedule changed")
        require(row["input_reservation_tokens"] == plan["pricing_contract"]["providers"][builder]["context_window_tokens"],
                "full-context reservation changed")
        reserve = pricing.reservation(builder)
        require(Decimal(row["reserve_usd"]) == reserve, "reservation amount changed")
        reserve_total += reserve
        check_file(row["wire"])
        require(path_safe(row["wire"]["path"]).is_relative_to(root/"wires"), "wire escaped canonical root")
        body = Path(row["wire"]["path"]).read_bytes()
        metadata = adapter.validate_wire(builder, condition, body)
        require(row.get("object_sha256", metadata["object_sha256"]) == metadata["object_sha256"],
                "normalized payload hash changed")
    require(Decimal(plan["maximum_reservation_usd"]) == reserve_total, "total reservation changed")
    if not fixture:
        require(reserve_total <= Decimal(plan["proposed_budget_usd"]), "twelve calls exceed new proposed cap")
    for builder in adapter.MODELS:
        adapter.verify_conditions(builder)
    _verify_sources(plan)
    # Freshness is checked separately on start and before every POST, allowing
    # an explicitly supplied same-contract renewal without cloning this plan.
    check_file(plan["pricing_evidence"])
    return plan


def create_plan(root, evidence, source_bindings, *, implementation_hashes=None,
                specification=None, fixture_only=False, proposed_budget_usd="3.25", extras=None):
    root = path_safe(Path(root).absolute())
    root.mkdir(parents=True, exist_ok=True)
    require(not (root/"plan.json").exists() and not (root/"enabled").exists(), "existing plan refused")
    wires_dir = root/"wires"; wires_dir.mkdir(exist_ok=False)
    wires = {}
    for builder in adapter.MODELS:
        adapter.verify_conditions(builder)
        for condition in adapter.CONDITIONS:
            body = adapter.wire_bytes(adapter.build_payload(builder, condition))
            p = wires_dir/f"{builder}_{condition}.json"
            with p.open("xb") as stream:
                stream.write(body)
            wires[(builder, condition)] = (descriptor(p), adapter.validate_wire(builder, condition, body))
    schedule = []
    for i, (builder, condition, repeat) in enumerate(EXPECTED_SCHEDULE, 1):
        wire, metadata = wires[(builder, condition)]
        schedule.append({"call_id": f"goal015c_call_{i:03d}", "builder": builder,
                         "condition": condition, "repeat_index": repeat, "wire": wire,
                         "object_sha256": metadata["object_sha256"],
                         "reserve_usd": str(pricing.reservation(builder)),
                         "input_reservation_tokens": pricing.contract()["providers"][builder]["context_window_tokens"]})
    plan = {"schema": "goal015c_plan_v1", "canonical_root": str(root),
            "api_allowed": False, "budget_approved": False, "scope_calls": 12,
            "concurrency": 1, "retries": 0, "timeout_seconds": 600,
            "proposed_budget_usd": proposed_budget_usd,
            "maximum_reservation_usd": str(sum((pricing.reservation(b) for b, _, _ in EXPECTED_SCHEDULE), Decimal(0))),
            "schedule": schedule, "pricing_contract": pricing.contract(),
            "pricing_evidence": evidence if isinstance(evidence, dict) else descriptor(evidence),
            "implementation_hashes": implementation_hashes or {}, "source_bindings": source_bindings,
            "task_summary": adapter.TASK_SUMMARY, "expected_document": adapter.EXPECTED_DOCUMENT,
            "specification": specification, "fixture_only": fixture_only,
            "old_budget_credit_used_usd": "0", "old_parent_cost_usd_separate": "4.43765700"}
    require(not set(extras or {}) & set(plan), "extras cannot override fixed contract")
    plan.update(extras or {})
    plan["plan_id"] = plan_identity(plan)
    pricing.write_new(root/"plan.json", plan)
    return plan
