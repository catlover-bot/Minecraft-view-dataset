import copy
from decimal import Decimal

import pytest

from offline_evaluation.experiments.goal015d import mini_results as m


def fixture_documents():
    A, B, C = m.CONDITIONS
    sequence = [("openai", A), ("anthropic", C), ("openai", B), ("anthropic", A),
                ("openai", C), ("anthropic", B), ("anthropic", B), ("openai", C),
                ("anthropic", A), ("openai", B), ("anthropic", C), ("openai", A)]
    rows, schedule, calls, seen = [], [], [], {}
    for index, (builder, condition) in enumerate(sequence, 1):
        seen[builder, condition] = seen.get((builder, condition), 0) + 1
        identity = dict(call_id=f"goal015c_call_{index:03d}", builder=builder,
                        condition=condition, repeat_index=seen[builder, condition])
        success = condition == B or (builder == "openai" and condition == C)
        json_failure = builder == "anthropic" and condition == C
        desc = {"path": identity["call_id"], "bytes": 0, "sha256": "synthetic"}
        row = {**identity, "wire": desc, "raw": desc, "response_text_path": desc,
               "state": "SETTLED", "send_status": "POST_RETURNED", "known_cost_usd": "0.001",
               "usage_known": True, "known_cost_verified_against_raw_usage": True,
               "response_model": "fixture", "completed_response": True,
               "stop_reason": "end_turn" if builder == "anthropic" else None,
               "raw_status": "completed" if builder == "openai" else None,
               "protocol_valid": not json_failure, "task_pass": success,
               "operation_sequence_conforms": success, "operation_count": None if json_failure else 3,
               "carve_count": None if json_failure else 1 if success else 3,
               "set_count": None if json_failure else int(success),
               "fill_count": None if json_failure else int(success),
               "failure_kind": "json_failure" if json_failure else None}
        rows.append(row)
        calls.append(copy.deepcopy(row))
        schedule.append({**identity, "wire": desc})
    binding = {"plan_id": m.PLAN_ID}
    groups = []
    for builder in ("openai", "anthropic"):
        for condition in m.CONDITIONS:
            selected = [r for r in rows if (r["builder"], r["condition"]) == (builder, condition)]
            groups.append({"builder": builder, "condition": condition, "planned": 2,
                           "strict_success": sum(r["task_pass"] for r in selected),
                           "protocol_valid": sum(r["protocol_valid"] for r in selected),
                           "operation_sequence_conforming": sum(r["operation_sequence_conforms"] for r in selected),
                           "known_cost_usd": "0.002"})
    verification = {
        "status": "GITHUB_VERIFIED", "release_id": 418290999, "asset_id": 659057302,
        "draft": True, "uploaded": True,
        "archive_sha256": "05b5a005ed49dc982541f2f48e6a97e986e487ed77d7fd99fae0ea090a51e908",
        "restoration": {"archive_sha256": "05b5a005ed49dc982541f2f48e6a97e986e487ed77d7fd99fae0ea090a51e908",
                        "plan_id": m.PLAN_ID, "status": "VERIFIED", "evaluation_included": True}}
    return dict(plan={"plan_id": m.PLAN_ID, "schedule": schedule},
                ledger={"plan_id": m.PLAN_ID, "calls": calls},
                evaluation={"plan_binding": binding, "rows": rows},
                summary={"plan_binding": binding, "by_builder_condition": groups,
                         "all": {"known_cost_usd": "0.012", "planned": 12, "completed_response": 12,
                                 "strict_success": 6, "protocol_valid": 10}},
                verification=verification)


def test_saved_report_reads_only_two_failed_texts_and_never_repairs():
    visited = []
    def reader(desc):
        visited.append(desc["path"])
        return b'```json\n{"operations":[]}\n```'
    rows, result = m.summarize_saved(**fixture_documents(), text_reader=reader)
    assert visited == ["goal015c_call_002", "goal015c_call_011"]
    assert len(rows) == 12 and Decimal(result["actual_known_cost_usd"]) == Decimal("0.012")
    assert result["independent_synthetic_tasks"] == 1
    assert all(i["classification"] == "MARKDOWN_CODE_FENCE" and not i["inner_json_parsed_or_repaired"]
               for i in result["failure_text_inspection"])
    assert [rows[i]["task_pass"] for i in (1, 10)] == [False, False]


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["ledger"]["calls"].reverse(), "identity"),
    (lambda d: d["evaluation"]["rows"][0].update(known_cost_usd="0.1"), "source or state"),
    (lambda d: d["summary"]["by_builder_condition"][0].update(strict_success=2), "aggregation"),
    (lambda d: d["verification"].update(asset_id=1), "verification"),
])
def test_inconsistent_saved_evidence_stops_report(mutate, match):
    documents = fixture_documents()
    mutate(documents)
    with pytest.raises(ValueError, match=match):
        m.summarize_saved(**documents, text_reader=lambda d: b"```json\n{}\n```")


def test_failure_category_does_not_claim_unread_inner_json():
    result = m.classify_saved_failure("explanation\n{broken")
    assert result["classification"] == "OTHER_STRICT_JSON_FAILURE"
    assert result["inner_json_parsed_or_repaired"] is False
    assert m.classify_saved_failure("```json\n{broken\n```")["classification"] == "MARKDOWN_CODE_FENCE"


def test_text_hash_verified_before_decoding(tmp_path):
    path = tmp_path / "text"
    path.write_bytes(b"\xff")
    desc = m.descriptor(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        m.checked_bytes(desc)
