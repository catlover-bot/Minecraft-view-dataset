# Goal007 audit and reproduction

Status: source audit PASS as supplied in `render_context.json`; analysis label **fixed60 analysis with one operational missing observation**. Rendering consumes the frozen `results.json`; it does not rerun inference, the evaluator, API clients, key lookup, or GitHub writes.

## Sources and classification

`render_context.json` records source proofs, costs, unknown-usage holds, and the reproduction command supplied by integration. `rq2_matched_scores_audit.json` records the separate matched-score investigation. The source plans, immutable hashes, restored releases/assets, ledger attempts, raw/parsed/voxel references, evaluator caches, and final source inventories are documented by the integration MANIFEST and audit receipts, not inferred from successful figure rendering.

Final slots are separate from attempts: 960 selected slots; 961 fixed-scope attempts and 991 historical calls; zero new research requests. Global940 selects the explicitly approved manual retry while preserving original401 attempt provenance and hold. Global942 is operationally unmeasurable HTTP520, with null response, usage, and metrics. Completed-request status requires identified completion and known usage; a received HTTP response alone is insufficient. Model-output parse/validation/execution/truncation failures are retained. Only these known completed failures may receive zero in auxiliary bounded IoU. Failed MAE/repair stay null. Operational unknowns never enter failure0. Actual invoices remain unknown.

## Exact metrics and all denominators

The metric version is `server_v2.2_strict_translation`: aligned scores use the canonical all-translation strict evaluator, with no finite-radius substitution or GT correction. Resource-limit/GT failures remain separately classified and cannot be replaced by approximations. `per_group.csv` contains all primary and secondary metric-specific valid row/building denominators and equal-building means, including the 12 description-provider × Builder × O/J combinations. P/E remain one observation per building and Builder.

`denominators.json` is reproduced verbatim below so planned, completed, valid, model failure, unknown, operationally missing, failure0-eligible and comparison denominators remain distinct.

```json
{
  "attempt_history_counts": "Reported separately from the verified attempt audit; never inferred by counting final slots.",
  "comparisons": [
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 180,
      "common_valid_pair_count": 176,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:O:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 176,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:O:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 179,
      "common_valid_pair_count": 177,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:J:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 177,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:J:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 60,
      "common_valid_pair_count": 60,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:P:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 60,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:P:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 46,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 60,
      "common_valid_pair_count": 46,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:E:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 46,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 46,
      "family_id": "primary_builder_valid_pair",
      "id": "anthropic-minus-openai:E:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 179,
      "common_valid_pair_count": 175,
      "family_id": "secondary_lossless_J_minus_O",
      "id": "J-minus-O:openai:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 175,
      "family_id": "secondary_lossless_J_minus_O",
      "id": "J-minus-O:openai:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 180,
      "common_valid_pair_count": 179,
      "family_id": "secondary_lossless_J_minus_O",
      "id": "J-minus-O:anthropic:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 179,
      "family_id": "secondary_lossless_J_minus_O",
      "id": "J-minus-O:anthropic:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 60,
      "common_valid_pair_count": 60,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "P-minus-O:openai:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 60,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "P-minus-O:openai:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 55,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 60,
      "common_valid_pair_count": 55,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "E-minus-O:openai:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 55,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 55,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "E-minus-O:openai:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 60,
      "common_valid_pair_count": 60,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "P-minus-O:anthropic:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 60,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 60,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "P-minus-O:anthropic:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    },
    {
      "buildings": 46,
      "common_failure0_buildings": 60,
      "common_failure0_pair_count": 60,
      "common_valid_pair_count": 46,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "E-minus-O:anthropic:aligned_occupancy_iou",
      "metric": "aligned_occupancy_iou"
    },
    {
      "buildings": 46,
      "common_failure0_buildings": null,
      "common_failure0_pair_count": null,
      "common_valid_pair_count": 46,
      "family_id": "diagnostic_oracle_minus_original",
      "id": "E-minus-O:anthropic:aligned_dimension_mae",
      "metric": "aligned_dimension_mae"
    }
  ],
  "groups": [
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 180,
      "completed_request_outcome_slots": 180,
      "condition": "O",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 3,
      "failure_kind_counts": {
        "validation_failure": 3
      },
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 3,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 180,
      "provider": "openai",
      "selected_attempts": 180,
      "status_counts": {
        "failed": 3,
        "success": 177
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 177
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "O",
      "description_model": "openai",
      "failure0_definitive_model_failure_slots": 3,
      "failure_kind_counts": {
        "validation_failure": 3
      },
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 3,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "failed": 3,
        "success": 57
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 57
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "O",
      "description_model": "anthropic",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "O",
      "description_model": "gemini",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 179,
      "completed_request_outcome_slots": 179,
      "condition": "J",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 2,
      "failure_kind_counts": {
        "http_failure": 1,
        "validation_failure": 2
      },
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 2,
      "operational_unmeasurable_slots": 1,
      "planned_slots": 180,
      "provider": "openai",
      "selected_attempts": 180,
      "status_counts": {
        "failed": 2,
        "operational_unmeasurable": 1,
        "success": 177
      },
      "unknown_usage_slots": 1,
      "valid_only_aligned_iou_slots": 177
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 59,
      "completed_request_outcome_slots": 59,
      "condition": "J",
      "description_model": "openai",
      "failure0_definitive_model_failure_slots": 2,
      "failure_kind_counts": {
        "http_failure": 1,
        "validation_failure": 2
      },
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 2,
      "operational_unmeasurable_slots": 1,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "failed": 2,
        "operational_unmeasurable": 1,
        "success": 57
      },
      "unknown_usage_slots": 1,
      "valid_only_aligned_iou_slots": 57
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "J",
      "description_model": "anthropic",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "J",
      "description_model": "gemini",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "P",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "E",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 5,
      "failure_kind_counts": {
        "truncation": 5
      },
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 5,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "openai",
      "selected_attempts": 60,
      "status_counts": {
        "failed": 5,
        "success": 55
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 55
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 180,
      "completed_request_outcome_slots": 180,
      "condition": "O",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 1,
      "failure_kind_counts": {
        "validation_failure": 1
      },
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 1,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 180,
      "provider": "anthropic",
      "selected_attempts": 180,
      "status_counts": {
        "failed": 1,
        "success": 179
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 179
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "O",
      "description_model": "openai",
      "failure0_definitive_model_failure_slots": 1,
      "failure_kind_counts": {
        "validation_failure": 1
      },
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 1,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "failed": 1,
        "success": 59
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 59
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "O",
      "description_model": "anthropic",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "O",
      "description_model": "gemini",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 180,
      "completed_request_outcome_slots": 180,
      "condition": "J",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 180,
      "provider": "anthropic",
      "selected_attempts": 180,
      "status_counts": {
        "success": 180
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 180
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "J",
      "description_model": "openai",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "J",
      "description_model": "anthropic",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "J",
      "description_model": "gemini",
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "description_builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "P",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 0,
      "failure_kind_counts": {},
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 0,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "success": 60
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 60
    },
    {
      "alignment_resource_limit_slots": 0,
      "bounded_iou_eligible_slots": 60,
      "completed_request_outcome_slots": 60,
      "condition": "E",
      "description_model": null,
      "failure0_definitive_model_failure_slots": 14,
      "failure_kind_counts": {
        "truncation": 14
      },
      "group_level": "builder_condition",
      "gt_missing_slots": 0,
      "independent_buildings": 60,
      "known_completed_other_failure_slots": 0,
      "model_output_failure_slots": 14,
      "operational_unmeasurable_slots": 0,
      "planned_slots": 60,
      "provider": "anthropic",
      "selected_attempts": 60,
      "status_counts": {
        "failed": 14,
        "success": 46
      },
      "unknown_usage_slots": 0,
      "valid_only_aligned_iou_slots": 46
    }
  ],
  "overall": {
    "alignment_resource_limit_slots": 0,
    "bounded_iou_eligible_slots": 959,
    "completed_request_outcome_slots": 959,
    "failure0_definitive_model_failure_slots": 25,
    "failure_kind_counts": {
      "http_failure": 1,
      "truncation": 19,
      "validation_failure": 6
    },
    "gt_missing_slots": 0,
    "independent_buildings": 60,
    "known_completed_other_failure_slots": 0,
    "model_output_failure_slots": 25,
    "operational_unmeasurable_slots": 1,
    "planned_slots": 960,
    "selected_attempts": 960,
    "status_counts": {
      "failed": 25,
      "operational_unmeasurable": 1,
      "success": 934
    },
    "unknown_usage_slots": 1,
    "valid_only_aligned_iou_slots": 934
  }
}
```

## Inference

The unchanged `offline_evaluation/server_v2/analysis_v4.py::comparison_families()` defines 20 confirmatory contrasts in families 8/4/8. B−A signs are preserved. O/J pairs match within building and description provider; oracle contrasts average eligible O descriptions within building without replicating P/E. Common-valid differences are averaged within each building, and buildings receive equal weight. Independent N is at most60.

Seed 20261013; deterministic SHA256 counter RNG; building percentile bootstrap 2000 replicates; two-sided building sign-flip 10000 Monte Carlo replicates with plus-one correction, or the frozen exact sign enumeration when at most16 buildings. Holm adjustment is applied within each original family, retaining nonestimable comparisons in the planned multiplicity. `comparisons_20.csv` retains raw/adjusted p, CI, method, actual pair/building N, direction, and auxiliary bounded-IoU companion eligible N/effects. No supplementary significance tests are added by rendering.

`sensitivity_missing942.json` gives separate exploratory endpoint scenarios for hypothetical missing IoU in[0,1], with changes in pair/building availability. These are not observed zeros or confidence intervals; no sensitivity p-values are computed. MAE has no finite[0,1] bound. RQ2 status is NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES; no unsupported matching or quality correlation is introduced.

## Tables and figures

Four LaTeX fragments use `booktabs` and `graphicx`: `table_main.tex`, `table_ablation.tex`, `table_failures.tex`, `table_statistics.tex`. NA means unavailable/inapplicable, never zero. All twenty contrasts are retained.

Figures are generated with matplotlib Agg as 300-dpi PNG and vector SVG. `figure_main` and `figure_description_builders` use exact `per_group` means and annotate valid slot counts; they have no inferred confidence intervals. Their units are separated into IoU and MAE panels. `figure_comparisons` uses all20 stored paired effects and bootstrap95% intervals, split by metric; labels include common-valid pair/building counts. The dashed zero reference is not a significance threshold. `figure_sources.json` identifies every plotted statistic and its JSON field; `render_artifacts.json` records output SHA256 and size. No figures use raw response contents or external images.

## CPU reproduction

Run from `/root/workspace/Minecraft-view-dataset` with the recorded code commit and isolated analysis Python. The integration command is:

```sh
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B -m offline_evaluation.analysis.goal007.cli --audit-gate /root/minecraft-runs/goal007-analysis-20261016T130420Z-a8a7a399/audit/audit_gate.json --destination /root/minecraft-runs/goal007-analysis-replay-CHOOSE_NEW_UTC --code-commit 7500cea2160a8489f0e8bd06b6249d261226f943 --plot-python /root/workspace/Minecraft-view-dataset/.venv-server-v2-goal007-plots-20261016/bin/python --rq2-audit /root/workspace/Minecraft-view-dataset/reports/server_only/goal007/rq2_matched_scores_audit.json
```

Existing outputs are never overwritten. Rendering can also be replayed from the three stored JSON inputs into a fresh directory using `python -m offline_evaluation.analysis.goal007.render --results /absolute/results.json --context /absolute/render_context.json --rq2 /absolute/rq2_matched_scores_audit.json --output /absolute/new-output-directory`. Statistical reproduction additionally requires the immutable source audit gate and canonical slots; replaying a figure alone is not source revalidation.
