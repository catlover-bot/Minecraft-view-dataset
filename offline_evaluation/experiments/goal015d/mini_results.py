"""Authenticate and describe the saved Goal015C results; never rerate outputs."""
from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

SOURCE_ROOT = Path("/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03")
PLAN_ID = "95ef8e4c7e0d8e261087944374e1a690f32516caa91cf06fd95740a75b453503"
EXPECTED = {
    "plan": ("plan.json", "065fe859b152912e41e0b5ef8697fa11c6e2764a862a7f23528654a8e7a21995"),
    "ledger": ("enabled/ledger.json", "e9a9531f45e461480e7d95c718d881eb61d784deb165c59684e0da1dd5ef4e72"),
    "evaluation": ("evaluations/goal015c-20261020T105050090371Z-22988535/evaluation_12.json",
                   "3077b263565d959a7b50f6c6c8a272206c6e98b2ed1a96266fd6083f7436a479"),
    "summary": ("evaluations/goal015c-20261020T105050090371Z-22988535/summary.json",
                "2d33be610c9b548461dedc95ab59d9d4827305e38fe4df596aede3aab3388ea5"),
    "verification": ("github-verification/attempt-ba9b8731ab92473b87ca6aaf65a4e6a2/verification.json",
                     "4e317d0b2d6627561d8af64ae5e788f4821caff779f65856a386fcf94b74a3af"),
}
IDENTITY = ("call_id", "builder", "condition", "repeat_index")
CONDITIONS = ("NATIVE_CURRENT_ORDER", "NATIVE_ORIGINAL_PROPERTY_ORDER", "TEXT_JSON_INSTRUCTION_ONLY")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def descriptor(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked_bytes(desc):
    raw = Path(desc["path"]).read_bytes()
    require(len(raw) == desc["bytes"] and hashlib.sha256(raw).hexdigest() == desc["sha256"],
            "saved mini source hash mismatch")
    return raw


def classify_saved_failure(text):
    """Classify the intact strict failure; do not parse a stripped/repaired body."""
    stripped = text.strip()
    return {
        "classification": "MARKDOWN_CODE_FENCE" if stripped.startswith("```") else "OTHER_STRICT_JSON_FAILURE",
        "starts_with_markdown_fence": stripped.startswith("```"),
        "ends_with_markdown_fence": stripped.endswith("```"),
        "strict_failure_preserved": True,
        "fence_removed": False,
        "inner_json_parsed_or_repaired": False,
        "rerated": False,
    }


def summarize_saved(plan, ledger, evaluation, summary, verification, *, text_reader=checked_bytes):
    """Check all identities and formal counts before constructing public rows."""
    require(plan["plan_id"] == ledger["plan_id"] == PLAN_ID, "mini plan ID differs")
    require(len(plan["schedule"]) == len(ledger["calls"]) == len(evaluation["rows"]) == 12,
            "mini planned frame differs")
    require(evaluation["plan_binding"] == summary["plan_binding"], "mini evaluation/summary binding differs")
    require(evaluation["plan_binding"]["plan_id"] == PLAN_ID, "mini evaluation plan differs")
    require(verification["status"] == "GITHUB_VERIFIED"
            and verification["release_id"] == 418290999 and verification["asset_id"] == 659057302
            and verification["draft"] is True and verification["uploaded"] is True
            and verification["archive_sha256"] == "05b5a005ed49dc982541f2f48e6a97e986e487ed77d7fd99fae0ea090a51e908"
            and verification["restoration"]["archive_sha256"] == verification["archive_sha256"]
            and verification["restoration"]["plan_id"] == PLAN_ID
            and verification["restoration"]["status"] == "VERIFIED"
            and verification["restoration"]["evaluation_included"] is True,
            "mini GitHub saved verification differs")
    rows, failure_inspection = [], []
    for index, (record, saved, result) in enumerate(zip(plan["schedule"], ledger["calls"], evaluation["rows"]), 1):
        require(record["call_id"] == f"goal015c_call_{index:03d}", "mini call order differs")
        require(all(record[k] == saved[k] == result[k] for k in IDENTITY), "mini call identity differs")
        for key in ("wire", "raw", "response_text_path", "state", "send_status", "known_cost_usd"):
            require(saved.get(key) == result.get(key), "mini saved/result source or state differs: " + key)
        require(record["wire"] == result["wire"], "mini planned wire differs")
        require(saved["state"] == "SETTLED" and saved["usage_known"] is True
                and result["usage_known"] is True and result["known_cost_verified_against_raw_usage"] is True,
                "mini usage evidence differs")
        require(saved.get("response_model") == result["response_model"], "mini returned model differs")
        row = {k: result[k] for k in IDENTITY}
        for key in ("state", "send_status", "status", "response_model", "raw_status", "stop_reason",
                    "completed_response", "incomplete_details", "usage", "usage_known", "known_cost_usd",
                    "protocol_valid", "task_pass", "operation_sequence_conforms", "operation_count",
                    "set_count", "fill_count", "carve_count", "final_voxel_count", "truncation",
                    "touch_limit_rejected", "failure_kind"):
            row[key] = result.get(key)
        row["carve_only"] = (result["carve_count"] == result["operation_count"]
                             and result["operation_count"] > 0) if result["operation_count"] is not None else None
        row["raw_sha256"] = result["raw"]["sha256"]
        row["text_sha256"] = result["response_text_path"]["sha256"]
        row["finish"] = result["stop_reason"] or result["raw_status"]
        if index in (2, 11):
            require(result["builder"] == "anthropic" and result["condition"] == CONDITIONS[2]
                    and result["failure_kind"] == "json_failure"
                    and result["task_pass"] is False and result["protocol_valid"] is False,
                    "designated mini strict failure differs")
            text = text_reader(result["response_text_path"]).decode("utf-8")
            failure_inspection.append({"call_id": record["call_id"],
                                       "text": result["response_text_path"],
                                       **classify_saved_failure(text)})
        rows.append(row)
    groups = []
    for builder in ("openai", "anthropic"):
        for condition in CONDITIONS:
            selected = [r for r in rows if r["builder"] == builder and r["condition"] == condition]
            group = {
                "builder": builder, "condition": condition, "planned": len(selected),
                "task_pass": sum(r["task_pass"] is True for r in selected),
                "protocol_valid": sum(r["protocol_valid"] is True for r in selected),
                "operation_sequence_conforming": sum(r["operation_sequence_conforms"] is True for r in selected),
                "carve_only": sum(r["carve_only"] is True for r in selected),
                "known_cost_usd": str(sum((Decimal(r["known_cost_usd"]) for r in selected), Decimal(0))),
            }
            formal = next(g for g in summary["by_builder_condition"]
                          if (g["builder"], g["condition"]) == (builder, condition))
            require(group["planned"] == formal["planned"] == 2
                    and group["task_pass"] == formal["strict_success"]
                    and group["protocol_valid"] == formal["protocol_valid"]
                    and group["operation_sequence_conforming"] == formal["operation_sequence_conforming"]
                    and Decimal(group["known_cost_usd"]) == Decimal(formal["known_cost_usd"]),
                    "mini formal group aggregation differs")
            groups.append(group)
    cost = sum((Decimal(r["known_cost_usd"]) for r in rows), Decimal(0))
    require(cost == Decimal(summary["all"]["known_cost_usd"]), "mini formal total cost differs")
    require(summary["all"]["planned"] == summary["all"]["completed_response"] == 12
            and sum(r["task_pass"] is True for r in rows) == summary["all"]["strict_success"]
            and sum(r["protocol_valid"] is True for r in rows) == summary["all"]["protocol_valid"],
            "mini formal total counts differ")
    expected = {("openai", CONDITIONS[0]): 0, ("openai", CONDITIONS[1]): 2,
                ("openai", CONDITIONS[2]): 2, ("anthropic", CONDITIONS[0]): 0,
                ("anthropic", CONDITIONS[1]): 2, ("anthropic", CONDITIONS[2]): 0}
    require(all(g["task_pass"] == expected[g["builder"], g["condition"]] for g in groups),
            "observed mini result changes proposed canary rationale; STOP")
    return rows, {
        "schema": "goal015d_saved_mini_summary_v1", "status": "PASS_SAVED_RESULTS_CONFIRMED",
        "source_plan_id": PLAN_ID, "planned_calls": 12, "independent_synthetic_tasks": 1,
        "by_builder_condition": groups, "actual_known_cost_usd": str(cost),
        "parent_known_cost_usd_separate": "4.43765700", "goal015c_old_cap_usd_separate": "3.25",
        "goal015d_proposed_unapproved_cap_usd_separate": "5.00", "old_residual_calls_still_held": 8,
        "failure_text_inspection": failure_inspection,
        "existing_github_save": verification, "new_research_api_requests_sent": 0,
        "generation_repeated": False, "evaluation_rerun": False, "github_fetch_or_restore_repeated": False,
    }


def build_report(output_dir, source_root=SOURCE_ROOT):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources, documents = {}, {}
    for key, (relative, sha) in EXPECTED.items():
        path = Path(source_root) / relative
        desc = descriptor(path)
        require(desc["sha256"] == sha, "user-pinned mini artifact changed: " + key)
        sources[key] = desc
        documents[key] = json.loads(checked_bytes(desc))
    require(documents["evaluation"]["plan_binding"]["plan"] == sources["plan"]
            and documents["evaluation"]["plan_binding"]["ledger"] == sources["ledger"]
            and documents["summary"]["evaluation_artifact"] == sources["evaluation"],
            "user-pinned mini source graph differs")
    rows, summary = summarize_saved(**documents)
    summary["sources"] = sources
    for name, value in (("MINI_RESULTS_12.json", {"schema": "goal015d_saved_mini_rows_v1", "rows": rows}),
                        ("MINI_RESULTS_SUMMARY.json", summary)):
        with (output / name).open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    with (output / "MINI_RESULTS_12.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "NULL" if v is None else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                          if isinstance(v, (dict, list, bool)) else v for k, v in row.items()} for row in rows)
    return summary
