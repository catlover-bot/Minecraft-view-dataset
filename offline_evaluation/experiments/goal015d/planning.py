"""Immutable two-building, eight-pair development canary; no old continuation."""
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

from ..goal015c.planning import path_safe, descriptor, read_json, check_file, require, PlanError
from . import adapter, pricing

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_ROOT = Path("/root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac")
REGISTRY = ROOT/"reports/server_only/goal015d/EXECUTION_BINDING.json"
PARENT = Path("/root/minecraft-runs/goal015-20261017T170546Z")
CHILD = Path("/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79")
MINI = Path("/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03")
SCENES = ("buildings_100_v4_building_082", "buildings_100_v4_building_003")
A, B = adapter.CONDITIONS
EXPECTED_SCHEDULE = [
    (SCENES[0], "openai", A, 1), (SCENES[0], "openai", B, 1),
    (SCENES[0], "anthropic", B, 1), (SCENES[0], "anthropic", A, 1),
    (SCENES[0], "anthropic", A, 2), (SCENES[0], "anthropic", B, 2),
    (SCENES[0], "openai", B, 2), (SCENES[0], "openai", A, 2),
    (SCENES[1], "anthropic", A, 1), (SCENES[1], "anthropic", B, 1),
    (SCENES[1], "openai", B, 1), (SCENES[1], "openai", A, 1),
    (SCENES[1], "openai", A, 2), (SCENES[1], "openai", B, 2),
    (SCENES[1], "anthropic", B, 2), (SCENES[1], "anthropic", A, 2),
]


def plan_identity(plan):
    return hashlib.sha256(adapter.canonical_json({k: v for k, v in plan.items() if k != "plan_id"}).encode()).hexdigest()


def _verify_sources(plan):
    bindings = plan["source_bindings"]
    names = {"parent_plan", "parent_ledger", "parent_approval", "parent_authority", "parent_execution_lock",
             "child_plan", "child_required_absent", "mini_plan", "mini_ledger", "mini_approval",
             "mini_authority", "mini_execution_lock", "mini_completion_lock", "mini_completion_progress",
             "mini_evaluation", "mini_summary", "mini_verification"}
    require(set(bindings) == names, "source binding scope differs")
    for key in names - {"child_required_absent"}:
        check_file(bindings[key])
    for name in bindings["child_required_absent"]:
        require(not os.path.lexists(path_safe(name)), "old child state appeared")
    if plan.get("fixture_only") is True:
        return
    expected = {"parent_plan": PARENT/"plan.json", "parent_ledger": PARENT/"enabled/ledger.json",
                "parent_approval": PARENT/"enabled/approval.json", "parent_authority": PARENT/"enabled/authority.json",
                "parent_execution_lock": PARENT/"execution.lock", "child_plan": CHILD/"plan.json",
                "mini_plan": MINI/"plan.json", "mini_ledger": MINI/"enabled/ledger.json",
                "mini_approval": MINI/"enabled/approval.json", "mini_authority": MINI/"enabled/authority.json",
                "mini_execution_lock": MINI/"execution.lock", "mini_completion_lock": MINI/"completion.lock",
                "mini_completion_progress": MINI/"completion-progress.json",
                "mini_evaluation": MINI/"evaluations/goal015c-20261020T105050090371Z-22988535/evaluation_12.json",
                "mini_summary": MINI/"evaluations/goal015c-20261020T105050090371Z-22988535/summary.json",
                "mini_verification": MINI/"github-verification/attempt-ba9b8731ab92473b87ca6aaf65a4e6a2/verification.json"}
    entry = read_json(CANONICAL_ROOT/"ENTRY.json")["protected_files"]
    for key, path in expected.items():
        require(bindings[key]["path"] == str(path) and bindings[key] == entry[str(path)], "old source differs from entry")
    require(bindings["child_required_absent"] == [str(CHILD/n) for n in
            ("enabled", "execution.lock", "completion.lock")], "child absence scope differs")
    ledger = read_json(expected["parent_ledger"])
    states = [row["state"] for row in ledger["calls"]]
    require(len(states) == 80 and states.count("SETTLED") == 72 and states.count("NOT_SENT") == 8,
            "parent settlement changed")
    require(sum((Decimal(row.get("known_cost_usd") or "0") for row in ledger["calls"]), Decimal(0))
            == Decimal("4.43765700"), "parent cost changed")
    child = read_json(expected["child_plan"])
    require(child["api_allowed"] is False and child["budget_approved"] is False, "old child enabled")
    mini = read_json(expected["mini_ledger"])
    require(len(mini["calls"]) == 12 and all(r["state"] == "SETTLED" for r in mini["calls"]),
            "completed mini result changed")
    require(sum((Decimal(r["known_cost_usd"]) for r in mini["calls"]), Decimal(0))
            == Decimal("0.01865750"), "mini cost changed")
    proof = read_json(expected["mini_verification"])
    require(proof["status"] == "GITHUB_VERIFIED" and proof["draft"] is True
            and proof["release_id"] == 418290999 and proof["asset_id"] == 659057302,
            "mini saved proof changed")


def _verify_inputs(plan):
    require(set(plan["inputs"]) == set(SCENES) and set(plan["gt_sources"]) == set(SCENES),
            "exact two fixed scenes required")
    parent = read_json(plan["source_bindings"]["parent_plan"]["path"])
    fixture = plan.get("fixture_only") is True
    if not fixture:
        unique = list(dict.fromkeys(row["scene_id"] for row in parent["schedule"]))
        require(unique[:2] == list(SCENES), "parent first two scenes changed")
    for scene in SCENES:
        source = plan["inputs"][scene]
        check_file(source["input"])
        path = path_safe(source["input"]["path"])
        require(path.is_relative_to(Path(plan["canonical_root"])/"inputs"), "input escaped canonical root")
        body = path.read_bytes()
        text = body.decode("utf-8")
        adapter.validate_input(text)
        for builder in adapter.MODELS:
            adapter.verify_conditions(builder, text)
        if fixture:
            continue
        check_file(source["parent_input"])
        require(Path(source["parent_input"]["path"]).is_relative_to(PARENT/"private")
                and Path(source["parent_input"]["path"]).read_bytes() == body, "parent FULL input changed")
        require(plan["gt_sources"][scene] == parent["gt_sources"][scene], "GT provenance differs")
        require(set(source["parent_payloads"]) == set(adapter.MODELS), "both parent Builder inputs required")
        for builder, payload_desc in source["parent_payloads"].items():
            check_file(payload_desc)
            expected = [r["payload"] for r in parent["schedule"] if
                        r["scene_id"] == scene and r["builder"] == builder and r["condition"] == "FULL_RECORD"]
            require(len(expected) == 2 and all(d == payload_desc for d in expected), "parent FULL payload identity differs")
            payload = read_json(payload_desc["path"])
            require(adapter.input_from_payload(builder, payload).encode("utf-8") == body,
                    "FULL input string bytes differ from parent payload")
            require(adapter.canonical_json(payload) == adapter.canonical_json(adapter.build_payload(builder, A, text)),
                    "A differs from parent FULL request contract")


def verify_plan(plan_path):
    path = path_safe(Path(plan_path).absolute())
    plan = read_json(path)
    require(plan.get("schema") == "goal015d_plan_v1" and path.name == "plan.json", "wrong canary plan")
    require(plan.get("plan_id") == plan_identity(plan), "plan identity mismatch")
    root = path_safe(plan["canonical_root"])
    require(path == root/"plan.json", "copied plan refused")
    fixture = plan.get("fixture_only") is True
    if fixture:
        require(root != CANONICAL_ROOT and not (root.parent == Path("/root/minecraft-runs")
                and root.name.startswith("goal015d-")), "production root cannot be fixture")
    else:
        require(root == CANONICAL_ROOT, "unregistered canary root")
        registry = read_json(REGISTRY)
        require(registry["plan"] == descriptor(path) and registry["plan_id"] == plan["plan_id"]
                and registry["canonical_root"] == str(root), "registered plan binding mismatch")
    require(plan["api_allowed"] is False and plan["budget_approved"] is False, "plan flags must remain disabled")
    require(plan["scope_calls"] == 16 and plan["concurrency"] == 1 and plan["retries"] == 0
            and plan["timeout_seconds"] == 600, "canary scope changed")
    require(plan["pricing_contract"] == pricing.contract(), "canary contract changed")
    require(plan["old_budget_credit_used_usd"] == "0" and plan["old_parent_cost_usd_separate"] == "4.43765700"
            and plan["mini_known_cost_usd_separate"] == "0.01865750", "prior budget accounting changed")
    require(0 < Decimal(plan["proposed_budget_usd"]) <= Decimal("5.00"), "invalid proposed canary budget")
    if not fixture:
        require(plan["proposed_budget_usd"] == "5.00" and plan.get("implementation_hashes")
                and plan.get("specification"), "production cap/code/spec changed")
    for name, sha in plan.get("implementation_hashes", {}).items():
        require(descriptor(name)["sha256"] == sha, "implementation dependency changed")
    for key in ("specification", "prompt", "launcher", "mini_results"):
        if plan.get(key):
            check_file(plan[key])
    _verify_sources(plan)
    _verify_inputs(plan)
    require(len(plan["schedule"]) == 16, "exactly sixteen calls required")
    total = Decimal(0)
    unique_wires = set()
    for index, (row, expected) in enumerate(zip(plan["schedule"], EXPECTED_SCHEDULE, strict=True), 1):
        scene, builder, condition, repeat = expected
        pair_id = f"{scene}_{builder}_r{repeat}"
        require(row["call_id"] == f"goal015d_call_{index:03d}"
                and (row["scene_id"], row["builder"], row["condition"], row["repeat_index"]) == expected
                and row["pair_id"] == row["paired_block_id"] == pair_id
                and row["pair_index"] == (index-1)//2+1, "fixed canary schedule changed")
        require(row["input"] == plan["inputs"][scene]["input"], "row FULL input differs")
        require(row["input_reservation_tokens"] == plan["pricing_contract"]["providers"][builder]["context_window_tokens"],
                "full-context reservation changed")
        reserve = pricing.reservation(builder)
        require(Decimal(row["reserve_usd"]) == reserve, "reservation changed")
        total += reserve
        check_file(row["wire"])
        require(path_safe(row["wire"]["path"]).is_relative_to(root/"wires"), "wire escaped canonical root")
        text = Path(row["input"]["path"]).read_text()
        metadata = adapter.validate_wire(builder, condition, Path(row["wire"]["path"]).read_bytes(), text)
        require(row["object_sha256"] == metadata["object_sha256"]
                and row["ordered_sha256"] == metadata["ordered_sha256"], "wire representation binding changed")
        unique_wires.add(row["wire"]["path"])
    require(len(unique_wires) == 8 and Decimal(plan["maximum_reservation_usd"]) == total == Decimal("4.235008"),
            "eight wires and full sixteen reservations required")
    if not fixture:
        require(total <= Decimal(plan["proposed_budget_usd"]), "sixteen reservations exceed new cap")
    check_file(plan["pricing_evidence"])
    return plan


def create_plan(root, evidence, source_bindings, *, inputs, gt_sources, implementation_hashes=None,
                specification=None, fixture_only=False, proposed_budget_usd="5.00", extras=None):
    root = path_safe(Path(root).absolute())
    root.mkdir(parents=True, exist_ok=True)
    require(not (root/"plan.json").exists() and not (root/"enabled").exists(), "existing plan refused")
    (root/"wires").mkdir(exist_ok=False)
    wires = {}
    for scene in SCENES:
        text = Path(inputs[scene]["input"]["path"]).read_text()
        for builder in adapter.MODELS:
            adapter.verify_conditions(builder, text)
            for condition in adapter.CONDITIONS:
                body = adapter.wire_bytes(adapter.build_payload(builder, condition, text))
                path = root/"wires"/f"{scene}_{builder}_{condition}.json"
                with path.open("xb") as stream:
                    stream.write(body)
                wires[scene, builder, condition] = (descriptor(path), adapter.validate_wire(builder, condition, body, text))
    rows = []
    for index, (scene, builder, condition, repeat) in enumerate(EXPECTED_SCHEDULE, 1):
        wire, meta = wires[scene, builder, condition]
        pair = f"{scene}_{builder}_r{repeat}"
        rows.append({"call_id": f"goal015d_call_{index:03d}", "scene_id": scene, "builder": builder,
                     "condition": condition, "repeat_index": repeat, "pair_id": pair, "paired_block_id": pair,
                     "pair_index": (index-1)//2+1, "input": inputs[scene]["input"], "wire": wire,
                     "object_sha256": meta["object_sha256"], "ordered_sha256": meta["ordered_sha256"],
                     "reserve_usd": str(pricing.reservation(builder)),
                     "input_reservation_tokens": pricing.contract()["providers"][builder]["context_window_tokens"]})
    plan = {"schema": "goal015d_plan_v1", "canonical_root": str(root), "api_allowed": False,
            "budget_approved": False, "scope_calls": 16, "concurrency": 1, "retries": 0, "timeout_seconds": 600,
            "proposed_budget_usd": proposed_budget_usd, "maximum_reservation_usd": "4.235008",
            "schedule": rows, "inputs": inputs, "gt_sources": gt_sources, "pricing_contract": pricing.contract(),
            "pricing_evidence": evidence if isinstance(evidence, dict) else descriptor(evidence),
            "implementation_hashes": implementation_hashes or {}, "specification": specification,
            "source_bindings": source_bindings, "fixture_only": fixture_only, "old_budget_credit_used_usd": "0",
            "old_parent_cost_usd_separate": "4.43765700", "mini_known_cost_usd_separate": "0.01865750"}
    require(not set(extras or {}) & set(plan), "extras cannot override fixed contract")
    plan.update(extras or {})
    plan["plan_id"] = plan_identity(plan)
    pricing.write_new(root/"plan.json", plan)
    return plan
