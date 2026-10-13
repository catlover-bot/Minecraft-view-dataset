"""Single human launch command; preparation and preview never send requests."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from offline_evaluation.server_v2.generation import GenerationError, utc_now
from offline_evaluation.server_v2.providers import KEY_NAMES
from . import plan as planning


def confirmation_text(plan: dict) -> str:
    return f"APPROVE GOAL004 {plan['identity_hash'][:12]} 16 SENDS 0 RETRIES USD5 TOTAL"


def preview(plan: dict) -> dict:
    identity = plan["identity"]
    return {"status": plan["status"], "api_allowed": False,
        "plan_identity_hash": plan["identity_hash"], "protocol_id": identity["protocol_id"],
        "fixed_targets_in_send_order": [{k: row[k] for k in
            ("schedule_index", "scene_id", "condition", "provider", "model", "input_id", "input_sha256")}
            for row in identity["schedule"]],
        "schema_sha256": identity["schema_sha256"], "prompt_sha256": identity["prompt_sha256"],
        "implementation_hashes": identity["implementation_hashes"], "budgets": identity["budgets"],
        "request_timeout_seconds": identity.get("timeout_seconds"),
        "output_cap_cpu_evidence_sha256": identity.get("output_cap_cpu_evidence_sha256"),
        "prior_usage": identity["parent"]["usage"], "parent_ledger_sha256": identity["parent"]["ledger_sha256"],
        "remaining_theoretical_usd": 4.74142575,
        "reservation": identity["reservation_policy"],
        "pricing_verified_on": {p: value["verified_on"] for p, value in identity["pricing"].items()},
        "api_key_present": {p: bool(os.environ.get(name)) for p, name in KEY_NAMES.items()},
        "account_structured_output_acceptance": "NOT_RUN",
        "canonical_output_directory": identity["enabled_directory"],
        "authority_path": identity["authority_path"],
        "scope": "16 fixed inputs only; no retries, model fallback, target replacement or full960",
        "cost_semantics": "application conservative stopping limits; not an invoice or guaranteed completion cost"}


def _confirm_from_tty(expected: str) -> bool:
    """Never accept stdin, environment variables, command flags, or default Enter."""
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as terminal:
            terminal.write("\nThis starts paid research requests for the displayed scope.\n")
            terminal.write(f"Type exactly: {expected}\n> ")
            terminal.flush()
            return terminal.readline().rstrip("\r\n") == expected
    except (OSError, EOFError):
        return False


def launch(plan_path, *, preview_only: bool = False) -> int:
    from . import runner
    plan = planning.load_plan(plan_path)
    display = preview(plan)
    print(json.dumps(display, ensure_ascii=False, sort_keys=True, indent=2))
    if preview_only:
        print("PREVIEW_ONLY: additional research sends=0; approval not requested")
        return 0
    identity = plan["identity"]
    planning.verify_parent(identity["parent"])
    planning.validate_pricing(identity["pricing"], fresh=True)
    if not all(display["api_key_present"].values()):
        print("MISSING_API_KEY: set both keys in this parent terminal; values are never printed")
        return 3
    enabled = Path(identity["enabled_directory"])
    if enabled.exists():
        # Runner validates the immutable human approval and the shared authority.
        # A partial activation is not overwritten or repaired by this launcher.
        print("RESUME: original canonical ledger only; no repeated or uncertain input is resent")
    else:
        if not _confirm_from_tty(confirmation_text(plan)):
            print("NOT_APPROVED: additional research sends=0; no enabled run created")
            return 2
        approval = {"manual_confirmed": True, "plan_identity_hash": plan["identity_hash"],
            "enabled_directory": identity["enabled_directory"], "approved_at_utc": utc_now(),
            "confirmation_channel": "/dev/tty", "scope": "goal004_fixed16_resmoke_only"}
        runner.initialize_enabled(plan_path, approval=approval)
    result = runner.run(plan_path, permission=True)
    summary = runner.evaluate_and_summarize(plan_path)
    print(json.dumps({"generation": result, "evaluation_summary": summary},
                     ensure_ascii=False, sort_keys=True, indent=2))
    # Exit0 is reserved for completing all planned observations. It still does
    # not imply sixteen nonempty or accurate buildings.
    complete = summary.get("experiment_complete", result.get("experiment_complete", False))
    return 0 if complete else 4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="CPU-only: create a new disabled preparation")
    prepare.add_argument("--destination", required=True)
    prepare.add_argument("--pricing", required=True)
    prepare.add_argument("--commit", required=True)
    start = sub.add_parser("launch", help="Display fixed scope and request one /dev/tty paid approval")
    start.add_argument("--plan", required=True)
    start.add_argument("--preview", action="store_true", help="Zero-send preview; never opens approval")
    status = sub.add_parser("show", help="Read-only disabled plan display")
    status.add_argument("--plan", required=True)
    evaluate = sub.add_parser("evaluate", help="CPU-only evaluation of the canonical enabled run")
    evaluate.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = planning.prepare(args.destination, pricing_file=args.pricing, commit=args.commit)
            print(json.dumps(preview(result), ensure_ascii=False, indent=2))
            return 0
        if args.command == "show":
            print(json.dumps(preview(planning.load_plan(args.plan)), ensure_ascii=False, indent=2))
            return 0
        if args.command == "evaluate":
            from .runner import evaluate_and_summarize
            print(json.dumps(evaluate_and_summarize(args.plan), ensure_ascii=False, indent=2))
            return 0
        return launch(args.plan, preview_only=args.preview)
    except (GenerationError, ValueError) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
