import copy

import pytest

from offline_evaluation.server_v2.analysis_v4 import (
    attempted_failure, bounded_all_attempted_value, collect_planned_outcomes, comparison_families,
    paired_building_inference, summarize_v4,
)


def outcome(scene="s1", provider="openai", condition="O", model="openai", value=.5, status="success", **fields):
    row = {"scene_id": scene, "provider": provider, "condition": condition,
           "description_model": model if condition in {"O", "J"} else None, "status": status,
           "metrics": {"metric_version": "server_v2.2_strict_translation",
                       "aligned": {"occupancy_iou": value, "dimension_mae": 2., "total_normalized_repair_cost": 3.},
                       "absolute": {"occupancy_iou": value / 2, "dimension_mae": 2., "total_normalized_repair_cost": 3.}} if status == "success" else None}
    row.update(fields)
    return row


def result_with_id(summary, comparison_id):
    return next(r for f in summary["families"] for r in f["results"] if r["id"] == comparison_id)


def test_cluster_inference_gives_each_building_equal_weight_not_each_description():
    pairs = [{"scene_id": "a", "a": 0., "b": 1.} for _ in range(3)] + [{"scene_id": "b", "a": 1., "b": 0.}]
    result = paired_building_inference(pairs, bootstrap_replicates=100)
    assert result["buildings"] == 2 and result["paired_rows"] == 4
    assert result["mean_delta"] == 0. and result["p_two_sided"] == 1.
    assert result == paired_building_inference(pairs[::-1], bootstrap_replicates=100)


def test_exact_sign_flip_and_insufficient_cluster_results():
    pairs = [{"scene_id": str(i), "a": 0., "b": 1.} for i in range(3)]
    result = paired_building_inference(pairs, bootstrap_replicates=100)
    assert result["p_two_sided"] == .25 and result["ci95"] == [1., 1.]
    assert paired_building_inference(pairs[:1])["p_two_sided"] is None
    assert paired_building_inference([])["mean_delta"] is None


@pytest.mark.parametrize("failure_kind", ["parse_failure", "validation_failure", "execution_failure", "truncation"])
def test_definitive_output_failure_gets_iou_zero_but_no_mae_or_repair(failure_kind):
    row = outcome(status="failed", attempt_id="attempt", failure_kind=failure_kind, completed_request_outcome=True)
    assert attempted_failure(row) and bounded_all_attempted_value(row) == 0.
    with pytest.raises(ValueError, match="only for bounded IoU"):
        bounded_all_attempted_value(row, "aligned_dimension_mae")
    group = summarize_v4([row], smoke=True)["groups"][0]
    assert group["definitive_attempted_failure_rows"] == 1
    assert group["failure_rate"] == 1.
    assert group["valid_only_metrics"]["aligned_dimension_mae"]["equal_building_mean"] is None
    assert group["valid_only_metrics"]["aligned_total_normalized_repair_cost"]["equal_building_mean"] is None


@pytest.mark.parametrize("failure_kind", ["auth", "billing", "model_unavailable", "context", "missing_key", "transport", "unknown_usage", "response_parse_failure", None])
def test_operational_errors_are_excluded_and_separately_counted(failure_kind):
    row = outcome(status="failed", attempt_id="attempt", failure_kind=failure_kind, completed_request_outcome=True)
    assert not attempted_failure(row) and bounded_all_attempted_value(row) is None
    group = summarize_v4([row], smoke=True)["groups"][0]
    assert group["failure_rate"] is None and group["all_attempted_rows"] == 0
    assert group["operational_or_unmeasurable_attempt_rows"] == 1


@pytest.mark.parametrize("status", ["NOT_RUN", "budget_stop", "budget_exhausted", "missing_input", "GT_MISSING", "ALIGNMENT_RESOURCE_LIMIT", "reserved", "infeasible"])
def test_planned_stopped_or_unmeasurable_rows_never_get_failure_zero(status):
    row = outcome(status=status, attempted=True, completed_request_outcome=True, failure_kind="parse_failure")
    assert bounded_all_attempted_value(row) is None


def test_generic_failure_without_completed_response_evidence_is_unmeasurable():
    row = outcome(status="failed", attempt_id="attempt", failure_kind="parse_failure")
    assert bounded_all_attempted_value(row) is None


def test_common_valid_pairs_are_metric_specific_and_oracles_are_not_replicated():
    rows = []
    for scene in ("a", "b"):
        for model in ("openai", "anthropic", "gemini"):
            rows.append(outcome(scene=scene, model=model, value=.25))
            rows.append(outcome(scene=scene, provider="anthropic", model=model, value=.75))
        rows.append(outcome(scene=scene, condition="E", value=1.))
    # One paired outcome has occupancy but no MAE/repair. Its occupancy remains usable.
    rows[1]["metrics"]["aligned"]["dimension_mae"] = None
    rows[1]["metrics"]["aligned"]["total_normalized_repair_cost"] = None
    summary = summarize_v4(rows, bootstrap_replicates=100)
    iou = result_with_id(summary, "anthropic-minus-openai:O:aligned_occupancy_iou")
    mae = result_with_id(summary, "anthropic-minus-openai:O:aligned_dimension_mae")
    oracle = result_with_id(summary, "E-minus-O:openai:aligned_occupancy_iou")
    assert iou["buildings"] == mae["buildings"] == oracle["buildings"] == 2
    assert iou["common_valid_pair_count"] == 6 and mae["common_valid_pair_count"] == 5
    assert oracle["common_valid_pair_count"] == 2 and oracle["mean_delta"] == .75
    repair = next(r for r in summary["secondary_descriptive_common_valid_pairs"] if r["id"] == "anthropic-minus-openai:O:aligned_total_normalized_repair_cost")
    assert repair["common_valid_pair_count"] == 5
    assert [f["predeclared_size"] for f in summary["families"]] == [8, 4, 8]
    assert all(r["p_holm"] is None for f in summary["families"] for r in f["results"] if r["p_two_sided"] is None)


def test_predeclared_holm_family_retains_nonestimable_comparisons():
    rows = []
    for i in range(6):
        rows += [outcome(scene=str(i), value=0.), outcome(scene=str(i), provider="anthropic", value=1.)]
    summary = summarize_v4(rows, bootstrap_replicates=100)
    result = result_with_id(summary, "anthropic-minus-openai:O:aligned_occupancy_iou")
    assert result["p_two_sided"] == .03125
    assert result["p_holm"] == .25  # family contains all eight original contrasts
    assert [f["size"] for f in comparison_families()] == [8, 4, 8]


def test_retries_failed_metrics_or_oracle_replication_cannot_inflate_analysis():
    good = outcome()
    with pytest.raises(ValueError, match="duplicate final"):
        summarize_v4([good, copy.deepcopy(good)])
    bad = outcome(status="failed")
    bad["metrics"] = {"aligned": {"dimension_mae": 0}}
    with pytest.raises(ValueError, match="metrics must be null"):
        summarize_v4([bad])
    bad = outcome(condition="E")
    bad["description_model"] = "openai"
    with pytest.raises(ValueError, match="must not be replicated"):
        summarize_v4([bad])


def test_smoke_has_descriptive_counts_but_no_research_inference():
    summary = summarize_v4([outcome(), outcome(provider="anthropic")], smoke=True)
    assert summary["families"] == [] and summary["secondary_descriptive_common_valid_pairs"] == []
    assert summary["smoke_inference_suppressed"] is True
    assert summary["evaluation_version"] == "server_v2.2_strict_translation"
    assert summary["research_requests_sent_by_analysis"] == 0


@pytest.mark.parametrize("version", ["server_v2.1", None, "historical_arr"])
def test_bounded_or_unknown_evaluation_versions_cannot_be_reported_as_strict(version):
    row = outcome()
    row["metrics"]["metric_version"] = version
    with pytest.raises(ValueError, match="requires server_v2.2_strict_translation"):
        summarize_v4([row])


def prepared_comparison(root, *, buildings=60):
    """Disabled, synthetic preparation fixtures; no HTTP or paid approval."""
    from offline_evaluation.server_v2.cohort_v4 import digest, json_text
    from offline_evaluation.server_v2.generation import GenerationConfig, object_hash, prepare_generation, write_json
    from offline_evaluation.server_v2.io_utils import sha256
    root.mkdir()
    cohort = {"dataset_version": "v4", "provenance": "fixture_only", "scenes": []}
    cohort["cohort_sha256"] = digest(json_text(cohort).encode())
    write_json(root / "cohort.v4.json", cohort)
    planned = []
    for index in range(buildings):
        for condition in "OJPE":
            for model in ("openai", "anthropic", "gemini") if condition in "OJ" else (None,):
                input_id = f"fixture-{index}-{condition}-{model}"
                planned.append({"input_id":input_id, "scene_id":f"fixture-scene-{index}", "condition":condition,
                                "dataset_version":"v4", "cohort_sha256":cohort["cohort_sha256"],
                                "description_model":model, "input_text":"Saved fixture record", "status":"prepared"})
    planned[-1].update(status="infeasible", reasons=["INPUT_BYTE_LIMIT"], input_text=None)
    write_json(root / "inputs.evaluation.json", planned)
    records = [r for r in planned if r["status"] == "prepared"]
    directories = []
    for provider in ("openai", "anthropic"):
        directory = root / f"builder_{provider}"
        manifest = prepare_generation(records, GenerationConfig(provider=provider, model="fixture-model-disabled",
            api_allowed=False, provenance="new_implementation"), directory, commit="a" * 40)
        write_json(directory / "selection.json", {"split":"evaluation", "planned_inputs":len(planned),
            "prepared_inputs":len(records), "planned_inputs_sha256":sha256(root / "inputs.evaluation.json"),
            "cohort_sha256":cohort["cohort_sha256"], "infeasible_inputs":[planned[-1]],
            "generation_identity_hash":object_hash(manifest["identity"])})
        directories.append(directory)
    return directories


def test_complete_960_planned_outcomes_survive_missing_evaluations_and_infeasible_inputs(tmp_path):
    directories = prepared_comparison(tmp_path / "comparison")
    rows = collect_planned_outcomes(directories)
    assert len(rows) == 960 and sum(r["status"] == "NOT_RUN" for r in rows) == 958
    assert sum(r["status"] == "infeasible" for r in rows) == 2
    assert all(r["metrics"] is None and r["attempted"] is False for r in rows)
    summary = summarize_v4(rows)
    assert sum(g["planned_rows"] for g in summary["groups"]) == 960
    assert sum(g["all_attempted_rows"] for g in summary["groups"]) == 0
    assert all(g["failure_rate"] is None for g in summary["groups"])


def test_duplicate_or_missing_provider_preparations_cannot_be_pooled(tmp_path):
    directories = prepared_comparison(tmp_path / "comparison", buildings=1)
    with pytest.raises(ValueError, match="duplicate provider"):
        collect_planned_outcomes([*directories, directories[0]])
    with pytest.raises(ValueError, match="missing provider"):
        collect_planned_outcomes(directories[:1])


@pytest.mark.parametrize("tamper", ["selection", "cohort", "planned_inputs", "campaign_identity"])
def test_summary_requires_matching_selection_cohort_plan_and_campaign_provenance(tmp_path, tamper):
    from offline_evaluation.server_v2.generation import read_json, write_json
    directories = prepared_comparison(tmp_path / "comparison", buildings=1)
    expected = None
    if tamper == "campaign_identity":
        expected = {"openai":"wrong", "anthropic":"wrong"}
    else:
        name = {"selection":directories[0] / "selection.json", "cohort":directories[0].parent / "cohort.v4.json",
                "planned_inputs":directories[0].parent / "inputs.evaluation.json"}[tamper]
        data = read_json(name)
        if tamper == "selection": data["planned_inputs"] = 1
        elif tamper == "cohort": data["provenance"] = "changed"
        else: data[0]["condition"] = "changed"
        write_json(name, data)
    with pytest.raises(ValueError, match="mismatch"):
        collect_planned_outcomes(directories, expected_identities=expected)


def test_evaluated_not_run_inputs_and_missing_rows_are_merged_without_erasing_plan(tmp_path):
    from offline_evaluation.server_v2.generation import evaluate_generation, read_json, write_json
    directories = prepared_comparison(tmp_path / "comparison", buildings=1)
    evaluate_generation(directories[0], {}, alignment_mode="strict_translation")
    document = read_json(directories[0] / "evaluation.json")
    document["rows"] = document["rows"][:1]
    write_json(directories[0] / "evaluation.json", document)
    rows = collect_planned_outcomes(directories)
    assert len(rows) == 16 and sum(r["status"] == "NOT_RUN" for r in rows) == 14
    assert sum(r["status"] == "infeasible" for r in rows) == 2


@pytest.mark.parametrize("tamper", ["wrong_provider", "unknown_input", "duplicate_input", "forged_success", "wrong_identity", "bounded_mode"])
def test_evaluation_rows_cannot_override_input_attempt_or_evaluation_identity(tmp_path, tamper):
    from offline_evaluation.server_v2.generation import evaluate_generation, read_json, write_json
    directories = prepared_comparison(tmp_path / "comparison", buildings=1)
    evaluate_generation(directories[0], {}, alignment_mode="strict_translation")
    path = directories[0] / "evaluation.json"
    document = read_json(path)
    if tamper == "wrong_provider": document["rows"][0]["provider"] = "anthropic"
    elif tamper == "unknown_input": document["rows"][0]["input_id"] = "unknown"
    elif tamper == "duplicate_input": document["rows"].append(copy.deepcopy(document["rows"][0]))
    elif tamper == "forged_success": document["rows"][0]["status"] = "success"
    elif tamper == "wrong_identity": document["identity_hash"] = "wrong"
    else: document["alignment_mode"] = "bounded"
    write_json(path, document)
    with pytest.raises(ValueError):
        collect_planned_outcomes(directories)


def test_cli_summary_keeps_unexecuted_plan_and_obeys_cooperative_run_lock(tmp_path, capsys):
    from offline_evaluation.server_v2.cli import main
    from offline_evaluation.server_v2.generation import GenerationError, read_json, run_lock
    root = tmp_path / "comparison"
    prepared_comparison(root, buildings=1)
    assert main(["summarize-v4", "--run-dir", str(root)]) is None
    summary = read_json(root / "summary.v4.json")
    assert sum(g["planned_rows"] for g in summary["groups"]) == 16
    assert sum(g["all_attempted_rows"] for g in summary["groups"]) == 0
    capsys.readouterr()
    with run_lock(root), pytest.raises(GenerationError, match="run_locked"):
        main(["summarize-v4", "--run-dir", str(root)])
